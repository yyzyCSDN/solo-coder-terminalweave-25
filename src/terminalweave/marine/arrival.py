"""Arrival-window planning that chains the independent marine gate checks.

The terminal already judges tide, weather, channel, pilotage, tugs and berths
separately. :class:`ArrivalPlanner` evaluates every gate together for a
requested vessel call and produces a *queryable* arrival plan:

* a plan is only released when every gate passes and all required information
  is present (each gate fails closed);
* once a forecast or resource revision is supplied, only draft (unconfirmed)
  plans are recomputed; confirmed and executed records are immutable;
* every rejection or reschedule carries the gate findings that explain it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from terminalweave.core.clock import ensure_utc, iso_z
from terminalweave.core.errors import (
    ConflictError,
    NotFoundError,
    SafetyConstraintError,
    StateTransitionError,
    ValidationError,
)
from terminalweave.core.intervals import TimeInterval
from terminalweave.marine.berth import BerthPlanner, VesselCall
from terminalweave.marine.berth_windows import BerthWindowCalendar
from terminalweave.marine.channel import ChannelRoute, TransitRequirement
from terminalweave.marine.pilotage import PilotJob, PilotRoster
from terminalweave.marine.tide import TideCurve, UnderKeelRequirement
from terminalweave.marine.tugs import TugAllocator, TugRequirement
from terminalweave.marine.weather import (
    MarineWeatherLimits,
    WeatherSample,
    WeatherWindowChecker,
)

# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


class Gate(str, Enum):
    DATA = 'data'
    TIDE = 'tide'
    CHANNEL = 'channel'
    WEATHER = 'weather'
    PILOT = 'pilot'
    TUG = 'tug'
    BERTH = 'berth'


@dataclass(frozen=True)
class GateFinding:
    """One reason a gate rejected a specific attempt."""

    gate: str
    code: str
    detail: str = ''
    at: datetime | None = None

    def __post_init__(self) -> None:
        if self.at is not None:
            object.__setattr__(self, 'at', ensure_utc(self.at))

    def to_dict(self) -> dict[str, object]:
        return {
            'gate': self.gate,
            'code': self.code,
            'detail': self.detail,
            'at': iso_z(self.at) if self.at is not None else None,
        }


# ---------------------------------------------------------------------------
# Requests, context and policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArrivalRequest:
    call_id: str
    loa_m: float
    beam_m: float
    draft_m: float
    pilot_district: str
    tug_requirement: TugRequirement
    earliest_departure: datetime
    latest_departure: datetime
    preferred_berth_id: str | None = None
    berth_stay: timedelta = timedelta(hours=4)
    night_transit: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, 'earliest_departure', ensure_utc(self.earliest_departure))
        object.__setattr__(self, 'latest_departure', ensure_utc(self.latest_departure))
        if min(self.loa_m, self.beam_m, self.draft_m) <= 0:
            raise ValidationError('vessel dimensions must be positive')
        if self.latest_departure <= self.earliest_departure:
            raise ValidationError('latest departure must follow earliest departure')
        if self.berth_stay.total_seconds() <= 0:
            raise ValidationError('berth stay must be positive')


@dataclass(frozen=True)
class MarineForecast:
    """Immutable bundle of the independent marine judgments.

    ``None`` for a data source means the forecast does not provide it, which
    forces the corresponding gate to fail closed.
    """

    tide: TideCurve | None
    weather_samples: tuple[WeatherSample, ...]
    weather_limits: MarineWeatherLimits | None
    channel: ChannelRoute | None
    revision: str
    published_at: datetime
    minimum_depth_margin_m: float = 0.5
    beam_clearance_factor: float = 1.5

    def __post_init__(self) -> None:
        object.__setattr__(self, 'published_at', ensure_utc(self.published_at))
        if not self.revision.strip():
            raise ValidationError('forecast revision required')
        if self.minimum_depth_margin_m < 0 or self.beam_clearance_factor <= 0:
            raise ValidationError('invalid channel margins')


@dataclass(frozen=True)
class ArrivalPolicy:
    search_step: timedelta = timedelta(minutes=30)
    weather_gap_tolerance: timedelta = timedelta(hours=1)
    max_transit_hours: float = 12.0
    minimum_clearance_m: float = 1.0
    squat_allowance_m: float = 0.0
    wave_allowance_m: float = 0.0

    def __post_init__(self) -> None:
        if self.search_step.total_seconds() <= 0:
            raise ValidationError('search step must be positive')
        if self.weather_gap_tolerance.total_seconds() < 0:
            raise ValidationError('weather gap tolerance cannot be negative')
        if self.max_transit_hours <= 0 or self.minimum_clearance_m < 0:
            raise ValidationError('invalid arrival policy')
        if min(self.squat_allowance_m, self.wave_allowance_m) < 0:
            raise ValidationError('allowances cannot be negative')


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


class ArrivalState(str, Enum):
    PROPOSED = 'proposed'
    CONFIRMED = 'confirmed'
    REJECTED = 'rejected'
    RESCHEDULED = 'rescheduled'
    EXECUTED = 'executed'


@dataclass(frozen=True)
class PlannedResources:
    pilot_id: str
    tug_ids: tuple[str, ...]
    berth_id: str
    berth_start_m: float

    def to_dict(self) -> dict[str, object]:
        return {
            'pilot_id': self.pilot_id,
            'tug_ids': list(self.tug_ids),
            'berth_id': self.berth_id,
            'berth_start_m': self.berth_start_m,
        }


@dataclass
class ArrivalPlan:
    plan_id: str
    request: ArrivalRequest
    state: ArrivalState
    window: TimeInterval | None = None
    transit_hours: float = 0.0
    resources: PlannedResources | None = None
    findings: tuple[GateFinding, ...] = ()
    forecast_revision: str = ''
    decided_at: datetime | None = None
    history: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.decided_at is not None:
            self.decided_at = ensure_utc(self.decided_at)

    @property
    def locked(self) -> bool:
        return self.state in {ArrivalState.CONFIRMED, ArrivalState.EXECUTED}

    def to_dict(self) -> dict[str, object]:
        return {
            'plan_id': self.plan_id,
            'call_id': self.request.call_id,
            'state': self.state.value,
            'window': {
                'start': iso_z(self.window.start),
                'end': iso_z(self.window.end),
            }
            if self.window is not None
            else None,
            'transit_hours': self.transit_hours,
            'resources': self.resources.to_dict() if self.resources else None,
            'findings': [f.to_dict() for f in self.findings],
            'forecast_revision': self.forecast_revision,
            'decided_at': iso_z(self.decided_at) if self.decided_at is not None else None,
            'history': list(self.history),
        }


# ---------------------------------------------------------------------------
# Replanning report
# ---------------------------------------------------------------------------


class ReplanDecision(str, Enum):
    UNCHANGED = 'unchanged'
    RESCHEDULED = 'rescheduled'
    REJECTED = 'rejected'
    CONFIRMED = 'confirmed'
    EXECUTED = 'executed'


@dataclass(frozen=True)
class ReplanItem:
    plan_id: str
    call_id: str
    decision: str
    previous_window: TimeInterval | None
    window: TimeInterval | None
    findings: tuple[GateFinding, ...]

    def to_dict(self) -> dict[str, object]:
        def window(value: TimeInterval | None) -> dict[str, str] | None:
            return {'start': iso_z(value.start), 'end': iso_z(value.end)} if value else None

        return {
            'plan_id': self.plan_id,
            'call_id': self.call_id,
            'decision': self.decision,
            'previous_window': window(self.previous_window),
            'window': window(self.window),
            'findings': [f.to_dict() for f in self.findings],
        }


@dataclass(frozen=True)
class ReplanReport:
    revision: str
    items: tuple[ReplanItem, ...]

    def rejected(self) -> tuple[ReplanItem, ...]:
        return tuple(i for i in self.items if i.decision == ReplanDecision.REJECTED.value)

    def rescheduled(self) -> tuple[ReplanItem, ...]:
        return tuple(i for i in self.items if i.decision == ReplanDecision.RESCHEDULED.value)

    def unchanged(self) -> tuple[ReplanItem, ...]:
        return tuple(i for i in self.items if i.decision == ReplanDecision.UNCHANGED.value)

    def to_dict(self) -> dict[str, object]:
        return {'revision': self.revision, 'items': [i.to_dict() for i in self.items]}


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class ArrivalPlanner:
    """Chains the marine gates and holds the queryable plan book."""

    def __init__(
        self,
        forecast: MarineForecast,
        pilots: PilotRoster,
        tugs: TugAllocator,
        berths: dict[str, BerthPlanner],
        closures: BerthWindowCalendar | None = None,
        policy: ArrivalPolicy | None = None,
    ):
        if not berths:
            raise ValidationError('at least one berth planner is required')
        self.forecast = forecast
        self.pilots = pilots
        self.tugs = tugs
        self.berths = dict(berths)
        self.closures = closures or BerthWindowCalendar()
        self.policy = policy or ArrivalPolicy()
        self._plans: dict[str, ArrivalPlan] = {}

    # -- queries -----------------------------------------------------------

    def get_plan(self, plan_id: str) -> ArrivalPlan:
        try:
            return self._plans[plan_id]
        except KeyError as exc:
            raise NotFoundError(f'unknown plan {plan_id}') from exc

    def plan_for_call(self, call_id: str) -> ArrivalPlan | None:
        for plan in self._plans.values():
            if plan.request.call_id == call_id:
                return plan
        return None

    def plans(self, state: ArrivalState | None = None) -> tuple[ArrivalPlan, ...]:
        plans = self._plans.values()
        if state is not None:
            plans = (p for p in plans if p.state == state)
        return tuple(sorted(plans, key=lambda p: p.plan_id))

    def released_plans(self) -> tuple[ArrivalPlan, ...]:
        return tuple(
            sorted(
                (p for p in self._plans.values() if p.window is not None),
                key=lambda p: (p.window.start, p.plan_id),
            )
        )

    # -- submission --------------------------------------------------------

    def submit(self, request: ArrivalRequest) -> ArrivalPlan:
        if self.plan_for_call(request.call_id) is not None:
            raise ConflictError(f'call {request.call_id} already has a plan')
        if request.preferred_berth_id is not None and request.preferred_berth_id not in self.berths:
            raise ValidationError(f'unknown berth {request.preferred_berth_id}')
        plan = ArrivalPlan(
            plan_id=request.call_id,
            request=request,
            state=ArrivalState.REJECTED,
            decided_at=self.forecast.published_at,
        )
        self._plans[plan.plan_id] = plan
        self._evaluate(plan)
        return plan

    def confirm(self, plan_id: str) -> ArrivalPlan:
        plan = self.get_plan(plan_id)
        if plan.state == ArrivalState.CONFIRMED:
            raise ConflictError('plan already confirmed')
        if plan.state == ArrivalState.EXECUTED:
            raise ConflictError('plan already executed')
        if plan.state == ArrivalState.REJECTED or plan.window is None:
            raise StateTransitionError('a rejected plan cannot be confirmed')
        plan.state = ArrivalState.CONFIRMED
        plan.history.append(
            {
                'at': iso_z(self.forecast.published_at),
                'event': 'confirmed',
                'forecast_revision': plan.forecast_revision,
            }
        )
        return plan

    def execute(self, plan_id: str, at: datetime) -> ArrivalPlan:
        at = ensure_utc(at)
        plan = self.get_plan(plan_id)
        if plan.state != ArrivalState.CONFIRMED:
            raise StateTransitionError('only confirmed plans can be executed')
        if plan.resources is not None:
            self.pilots.release(plan.plan_id)
            self.tugs.release(*plan.resources.tug_ids)
        plan.state = ArrivalState.EXECUTED
        plan.history.append({'at': iso_z(at), 'event': 'executed'})
        return plan

    # -- revision / replanning --------------------------------------------

    def update_forecast(self, forecast: MarineForecast) -> ReplanReport:
        if forecast.revision == self.forecast.revision:
            raise ValidationError('forecast revision is unchanged')
        self.forecast = forecast
        return self.replan()

    def resource_changed(self) -> ReplanReport:
        """Replan after a roster, fleet or closure change against the same forecast."""
        return self.replan()

    def replan(self) -> ReplanReport:
        items: list[ReplanItem] = []
        ordered = sorted(
            self._plans.values(),
            key=lambda p: (p.request.earliest_departure, p.plan_id),
        )
        for plan in ordered:
            if plan.state in {ArrivalState.CONFIRMED, ArrivalState.EXECUTED}:
                decision = (
                    ReplanDecision.CONFIRMED
                    if plan.state == ArrivalState.CONFIRMED
                    else ReplanDecision.EXECUTED
                )
                items.append(
                    ReplanItem(
                        plan.plan_id, plan.request.call_id, decision.value,
                        plan.window, plan.window, (),
                    )
                )
                continue
            previous = plan.window
            self._release(plan)
            old_blockers = (
                self._attempt(plan.request, previous.start)[1]
                if previous is not None
                else ()
            )
            self._evaluate(plan)
            decision = self._decision(plan, previous, old_blockers)
            if decision == ReplanDecision.RESCHEDULED.value:
                plan.state = ArrivalState.RESCHEDULED
            elif decision == ReplanDecision.UNCHANGED.value:
                plan.state = ArrivalState.PROPOSED
            items.append(
                ReplanItem(
                    plan.plan_id,
                    plan.request.call_id,
                    decision,
                    previous,
                    plan.window,
                    tuple(plan.findings),
                )
            )
        return ReplanReport(self.forecast.revision, tuple(items))

    def _decision(
        self,
        plan: ArrivalPlan,
        previous: TimeInterval | None,
        old_blockers: tuple[GateFinding, ...],
    ) -> str:
        if plan.window is None:
            return ReplanDecision.REJECTED.value
        if previous is None or plan.window != previous:
            if previous is not None and not old_blockers:
                plan.findings = (
                    GateFinding(
                        Gate.DATA.value,
                        'window_unavailable',
                        'previous window is no longer available',
                        previous.start,
                    ),
                )
            elif previous is not None and old_blockers:
                plan.findings = (
                    GateFinding(
                        Gate.TIDE.value,
                        'window_unsafe',
                        'previous window no longer satisfies the safety gates',
                        previous.start,
                    ),
                ) + old_blockers
            return ReplanDecision.RESCHEDULED.value
        return ReplanDecision.UNCHANGED.value

    # -- horizon coverage (fail closed) ------------------------------------

    def _horizon_end(self, request: ArrivalRequest) -> datetime:
        return request.latest_departure + timedelta(hours=self.policy.max_transit_hours)

    def _horizon_findings(self, request: ArrivalRequest) -> tuple[GateFinding, ...]:
        findings: list[GateFinding] = []
        horizon_end = self._horizon_end(request)
        if self.forecast.tide is None:
            findings.append(GateFinding(Gate.TIDE.value, 'forecast_missing', 'no tide forecast published'))
        elif request.earliest_departure < self.forecast.tide.start or horizon_end > self.forecast.tide.end:
            findings.append(
                GateFinding(
                    Gate.TIDE.value,
                    'forecast_gap',
                    f'tide forecast covers {iso_z(self.forecast.tide.start)}..{iso_z(self.forecast.tide.end)}',
                )
            )
        if self.forecast.channel is None:
            findings.append(GateFinding(Gate.CHANNEL.value, 'route_missing', 'no channel route configured'))
        if self.forecast.weather_limits is None or not self.forecast.weather_samples:
            findings.append(GateFinding(Gate.WEATHER.value, 'forecast_missing', 'no weather forecast published'))
        else:
            ordered = sorted(self.forecast.weather_samples, key=lambda s: s.at)
            if ordered[0].at > request.earliest_departure or ordered[-1].at < horizon_end:
                findings.append(
                    GateFinding(
                        Gate.WEATHER.value,
                        'forecast_gap',
                        'weather forecast does not cover the whole transit horizon',
                    )
                )
            else:
                for previous, current in zip(ordered, ordered[1:]):
                    if current.at - previous.at > self.policy.weather_gap_tolerance:
                        findings.append(
                            GateFinding(
                                Gate.WEATHER.value,
                                'forecast_gap',
                                'gap between weather samples exceeds tolerance',
                                current.at,
                            )
                        )
                        break
        if request.pilot_district not in {p.district for p in self.pilots.pilots.values()}:
            findings.append(
                GateFinding(
                    Gate.PILOT.value,
                    'district_missing',
                    f'no pilot roster for district {request.pilot_district}',
                )
            )
        if not self.tugs.tugs:
            findings.append(GateFinding(Gate.TUG.value, 'fleet_missing', 'no tug fleet configured'))
        if not self.berths:
            findings.append(GateFinding(Gate.BERTH.value, 'berth_missing', 'no berths configured'))
        return tuple(findings)

    # -- evaluation --------------------------------------------------------

    def _evaluate(self, plan: ArrivalPlan) -> None:
        request = plan.request
        horizon_findings = self._horizon_findings(request)
        if horizon_findings:
            self._apply_rejection(plan, horizon_findings)
            return
        ukc = self._ukc_requirement(request)
        cursor = request.earliest_departure
        while cursor <= request.latest_departure:
            transit_hours, findings = self._attempt(request, cursor, ukc)
            if not findings:
                self._apply_success(
                    plan,
                    TimeInterval(cursor, cursor + timedelta(hours=transit_hours)),
                    transit_hours,
                    ArrivalState.PROPOSED,
                )
                return
            cursor += self.policy.search_step
        self._apply_rejection(plan, self._aggregate_findings(request, ukc))

    def _attempt(
        self,
        request: ArrivalRequest,
        at: datetime,
        ukc: UnderKeelRequirement | None = None,
    ) -> tuple[float, tuple[GateFinding, ...]]:
        """Evaluate every gate for one departure; returns (transit hours, findings)."""
        ukc = ukc or self._ukc_requirement(request)
        findings: list[GateFinding] = []
        findings.append(self._tide_blocker(request, at, ukc))
        transit_hours, transit_finding = self._transit(request, at)
        if transit_finding is not None:
            findings.append(transit_finding)
        findings.append(self._weather_blocker(request, at))
        if transit_hours > 0:
            findings.append(self._pilot_blocker(request, at, transit_hours))
        findings.append(self._tug_blocker(request, at))
        findings.append(self._berth_blocker(request, at))
        return transit_hours, tuple(f for f in findings if f is not None)

    def _ukc_requirement(self, request: ArrivalRequest) -> UnderKeelRequirement:
        shallowest = (
            min(leg.chart_depth_m for leg in self.forecast.channel.legs)
            if self.forecast.channel is not None
            else 1.0
        )
        return UnderKeelRequirement(
            shallowest,
            request.draft_m,
            self.policy.minimum_clearance_m,
            self.policy.squat_allowance_m,
            self.policy.wave_allowance_m,
        )

    def _tide_blocker(
        self, request: ArrivalRequest, at: datetime, ukc: UnderKeelRequirement
    ) -> GateFinding | None:
        if self.forecast.tide is None:
            return GateFinding(Gate.TIDE.value, 'forecast_missing', 'no tide forecast published', at)
        try:
            if not self.forecast.tide.is_safe(at, ukc):
                return GateFinding(
                    Gate.TIDE.value,
                    'insufficient_ukc',
                    f"clearance {self.forecast.tide.clearance_m(at, ukc):.2f}m below {ukc.minimum_clearance_m:.2f}m",
                    at,
                )
        except SafetyConstraintError:
            return GateFinding(Gate.TIDE.value, 'forecast_gap', 'tide forecast does not cover attempt', at)
        return None

    def _transit(
        self, request: ArrivalRequest, departure: datetime
    ) -> tuple[float, GateFinding | None]:
        """Walk the legs sequentially, evaluating channel geometry at the ETA of each leg."""
        if self.forecast.channel is None:
            return 0.0, GateFinding(Gate.CHANNEL.value, 'route_missing', 'no channel route configured', departure)
        if self.forecast.tide is None:
            return 0.0, GateFinding(Gate.TIDE.value, 'forecast_missing', 'no tide forecast published', departure)
        total_hours = 0.0
        cursor = departure
        tide_by_leg: dict[str, float] = {}
        for leg in self.forecast.channel.legs:
            try:
                tide_by_leg[leg.leg_id] = self.forecast.tide.height_at(cursor)
            except SafetyConstraintError:
                return 0.0, GateFinding(
                    Gate.TIDE.value, 'forecast_gap', 'tide forecast does not cover transit', cursor
                )
            requirement = TransitRequirement(
                request.draft_m,
                request.beam_m,
                leg.speed_limit_kn,
                self.forecast.minimum_depth_margin_m,
                self.forecast.beam_clearance_factor,
            )
            try:
                results = ChannelRoute([leg]).evaluate(requirement, tide_by_leg)
            except SafetyConstraintError as exc:
                message = str(exc)
                if 'width' in message:
                    return 0.0, GateFinding(Gate.CHANNEL.value, 'insufficient_width', message, cursor)
                return 0.0, GateFinding(Gate.CHANNEL.value, 'insufficient_depth', message, cursor)
            total_hours += sum(r.hours for r in results)
            cursor += timedelta(hours=sum(r.hours for r in results))
        return total_hours, None

    def _weather_blocker(self, request: ArrivalRequest, at: datetime) -> GateFinding | None:
        if self.forecast.weather_limits is None or not self.forecast.weather_samples:
            return GateFinding(Gate.WEATHER.value, 'forecast_missing', 'no weather forecast published', at)
        ordered = sorted(self.forecast.weather_samples, key=lambda s: s.at)
        if at < ordered[0].at or at > ordered[-1].at:
            return GateFinding(Gate.WEATHER.value, 'forecast_gap', 'weather forecast does not cover attempt', at)
        sample = next((s for s in ordered if s.at >= at), ordered[-1])
        checker = WeatherWindowChecker(self.forecast.weather_limits)
        if not checker.acceptable(sample):
            return GateFinding(
                Gate.WEATHER.value,
                'conditions_exceed_limits',
                ', '.join(checker.reasons(sample)),
                sample.at,
            )
        return None

    def _pilot_blocker(self, request: ArrivalRequest, at: datetime, transit_hours: float) -> GateFinding | None:
        night = request.night_transit or (at.hour < 6 or at.hour >= 20)
        job = PilotJob(
            self._plan_id_for(request),
            request.pilot_district,
            request.loa_m,
            TimeInterval(at, at + timedelta(hours=max(2.0, transit_hours))),
            night=night,
        )
        reasons: set[str] = set()
        matched = False
        for pilot in self.pilots.pilots.values():
            if pilot.district != request.pilot_district:
                continue
            matched = True
            if request.loa_m > pilot.max_loa_m:
                reasons.add('loa')
            if job.night and not pilot.night_qualified:
                reasons.add('night_qualification')
            if job.interval.duration > pilot.max_shift:
                reasons.add('shift_limit')
            expanded = TimeInterval(job.interval.start - self.pilots.rest, job.interval.end + self.pilots.rest)
            if any(
                a.pilot_id == pilot.pilot_id and expanded.overlaps(a.interval)
                for a in self.pilots.assignments
            ):
                reasons.add('rest_conflict')
        if not matched:
            return GateFinding(
                Gate.PILOT.value,
                'district_missing',
                f'no pilot roster for district {request.pilot_district}',
                at,
            )
        if reasons:
            return GateFinding(Gate.PILOT.value, 'no_qualified_pilot', ', '.join(sorted(reasons)), at)
        return None

    def _tug_blocker(self, request: ArrivalRequest, at: datetime) -> GateFinding | None:
        available = [t for t in self.tugs.tugs.values() if t.tug_id not in self.tugs.busy]
        pull = sum(t.bollard_pull_t for t in available)
        azimuthing = sum(1 for t in available if t.azimuthing)
        firefighting = any(t.firefighting for t in available)
        req = request.tug_requirement
        if (
            len(available) < req.minimum_count
            or pull < req.minimum_total_pull_t
            or azimuthing < req.require_azimuthing
            or (req.require_firefighting and not firefighting)
        ):
            return GateFinding(
                Gate.TUG.value,
                'requirement_unmet',
                f'{len(available)} free tugs, {pull:.0f}t combined pull',
                at,
            )
        return None

    def _berth_order(self, request: ArrivalRequest) -> list[str]:
        ids = list(self.berths)
        if request.preferred_berth_id is not None:
            ids.sort(key=lambda b: (b != request.preferred_berth_id, b))
        return ids

    def _berth_blocker(self, request: ArrivalRequest, at: datetime) -> GateFinding | None:
        stay = request.berth_stay
        vessel = self._vessel_call(request, at)
        unavailable: list[str] = []
        for berth_id in self._berth_order(request):
            planner = self.berths[berth_id]
            free = self.closures.available(berth_id, TimeInterval(at, at + stay))
            if not any(interval.duration >= stay for interval in free):
                unavailable.append(f'{berth_id}:closure')
                continue
            if vessel.draft_m > planner.berth.max_draft_m:
                unavailable.append(f'{berth_id}:draft_limit')
                continue
            if planner.first_fit_position(vessel) is None:
                unavailable.append(f'{berth_id}:occupied')
        if len(unavailable) == len(self.berths):
            return GateFinding(Gate.BERTH.value, 'no_berth', ', '.join(unavailable), at)
        return None

    def _aggregate_findings(self, request: ArrivalRequest, ukc: UnderKeelRequirement) -> tuple[GateFinding, ...]:
        unique: dict[tuple[str, str], GateFinding] = {}
        cursor = request.earliest_departure
        while cursor <= request.latest_departure:
            for finding in self._attempt(request, cursor, ukc)[1]:
                unique.setdefault((finding.gate, finding.code), finding)
            cursor += self.policy.search_step
        if not unique:
            unique[(Gate.DATA.value, 'no_window')] = GateFinding(
                Gate.DATA.value, 'no_window', 'no feasible departure inside the requested horizon'
            )
        return tuple(unique[key] for key in sorted(unique))

    # -- resource commit / release ----------------------------------------

    def _plan_id_for(self, request: ArrivalRequest) -> str:
        plan = self.plan_for_call(request.call_id)
        return plan.plan_id if plan is not None else request.call_id

    def _vessel_call(self, request: ArrivalRequest, departure: datetime) -> VesselCall:
        return VesselCall(
            self._plan_id_for(request),
            request.loa_m,
            request.beam_m,
            departure,
            departure + request.berth_stay,
            request.draft_m,
        )

    def _commit(self, request: ArrivalRequest, window: TimeInterval, transit_hours: float) -> PlannedResources:
        night = request.night_transit or (window.start.hour < 6 or window.start.hour >= 20)
        job = PilotJob(
            self._plan_id_for(request),
            request.pilot_district,
            request.loa_m,
            TimeInterval(window.start, window.start + timedelta(hours=max(2.0, transit_hours))),
            night=night,
        )
        assignment = self.pilots.assign(job)
        try:
            tugs = self.tugs.allocate(request.tug_requirement)
        except Exception:
            self.pilots.release(job.job_id)
            raise
        try:
            vessel = self._vessel_call(request, window.start)
            berth_assignment = None
            for berth_id in self._berth_order(request):
                planner = self.berths[berth_id]
                stay = request.berth_stay
                free = self.closures.available(berth_id, TimeInterval(window.start, window.start + stay))
                if not any(interval.duration >= stay for interval in free):
                    continue
                if planner.first_fit_position(vessel) is None:
                    continue
                berth_assignment = planner.assign(vessel)
                break
            if berth_assignment is None:
                raise ConflictError('no conflict-free berth position is available')
        except Exception:
            self.tugs.release(*(t.tug_id for t in tugs))
            self.pilots.release(job.job_id)
            raise
        return PlannedResources(
            assignment.pilot_id,
            tuple(t.tug_id for t in tugs),
            berth_assignment.berth_id,
            berth_assignment.start_m,
        )

    def _release(self, plan: ArrivalPlan) -> None:
        if plan.resources is None:
            return
        self.pilots.release(plan.plan_id)
        self.tugs.release(*plan.resources.tug_ids)
        planner = self.berths.get(plan.resources.berth_id)
        if planner is not None:
            planner.unassign(plan.plan_id)
        plan.resources = None

    def _apply_success(self, plan: ArrivalPlan, window: TimeInterval,
                       transit_hours: float, state: ArrivalState) -> None:
        resources = self._commit(plan.request, window, transit_hours)
        plan.window = window
        plan.transit_hours = transit_hours
        plan.resources = resources
        plan.findings = ()
        plan.forecast_revision = self.forecast.revision
        plan.decided_at = self.forecast.published_at
        plan.state = state
        plan.history.append(
            {
                'at': iso_z(self.forecast.published_at),
                'event': plan.state.value,
                'window_start': iso_z(window.start),
                'window_end': iso_z(window.end),
                'forecast_revision': self.forecast.revision,
            }
        )

    def _apply_rejection(self, plan: ArrivalPlan, findings: tuple[GateFinding, ...]) -> None:
        previous = plan.window
        self._release(plan)
        plan.window = None
        plan.transit_hours = 0.0
        plan.findings = findings
        plan.forecast_revision = self.forecast.revision
        plan.decided_at = self.forecast.published_at
        plan.state = ArrivalState.REJECTED
        plan.history.append(
            {
                'at': iso_z(self.forecast.published_at),
                'event': 'rejected',
                'forecast_revision': self.forecast.revision,
                'reasons': [f.to_dict() for f in findings],
                'previous_window': {
                    'start': iso_z(previous.start),
                    'end': iso_z(previous.end),
                }
                if previous is not None
                else None,
            }
        )
