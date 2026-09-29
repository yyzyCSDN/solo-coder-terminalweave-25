"""Queryable port-entry planning that chains the independent marine checks.

The terminal already judges tide, weather, channel geometry, pilotage, tug
capacity and berth availability separately. :class:`PortEntryPlanner` chains
them into a single clearance decision. The gate fails closed: missing data or
any unsatisfied safety condition keeps the vessel out and every reason is
reported. Forecast or resource changes only replan unconfirmed entries;
confirmed and executed records are immutable.
"""
from __future__ import annotations
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from terminalweave.core.clock import ensure_utc, iso_z
from terminalweave.core.errors import (
    ValidationError,
    NotFoundError,
    StateTransitionError,
    SafetyConstraintError,
    CapacityError,
    ConflictError,
)
from terminalweave.core.intervals import TimeInterval
from terminalweave.marine.tide import TideCurve, UnderKeelRequirement
from terminalweave.marine.weather import WeatherSample, MarineWeatherLimits, WeatherWindowChecker
from terminalweave.marine.channel import ChannelRoute, TransitRequirement
from terminalweave.marine.pilotage import Pilot, PilotJob, PilotAssignment, PilotRoster
from terminalweave.marine.tugs import Tug, TugRequirement, TugAllocator
from terminalweave.marine.berth import Berth, VesselCall, BerthPlanner
from terminalweave.marine.berth_windows import BerthWindowCalendar, BerthClosure

# Reason codes returned with every rejection or postponement.
TIDE_MISSING = 'tide_forecast_missing'
TIDE_UKC = 'tide_under_keel_clearance'
WEATHER_MISSING = 'weather_forecast_missing'
WEATHER_GAP = 'weather_forecast_gap'
WEATHER_WIND = 'weather_wind'
WEATHER_GUST = 'weather_gust'
WEATHER_VISIBILITY = 'weather_visibility'
WEATHER_WAVE = 'weather_wave'
WEATHER_LIGHTNING = 'weather_lightning'
CHANNEL_ROUTE_MISSING = 'channel_route_missing'
CHANNEL_DEPTH = 'channel_depth'
CHANNEL_WIDTH = 'channel_width'
PILOT_ROSTER_MISSING = 'pilot_roster_missing'
PILOT_UNAVAILABLE = 'pilot_unavailable'
TUG_FLEET_MISSING = 'tug_fleet_missing'
TUG_UNAVAILABLE = 'tug_unavailable'
BERTH_MISSING = 'berth_missing'
BERTH_DRAFT = 'berth_draft_limit'
BERTH_OCCUPIED = 'berth_occupied'
BERTH_CLOSED = 'berth_closed'

_WEATHER_CODE = {
    'sustained wind': WEATHER_WIND,
    'wind gust': WEATHER_GUST,
    'visibility': WEATHER_VISIBILITY,
    'wave height': WEATHER_WAVE,
    'lightning': WEATHER_LIGHTNING,
}


class PlanState(str, Enum):
    PROPOSED = 'proposed'
    CONFIRMED = 'confirmed'
    EXECUTED = 'executed'


@dataclass(frozen=True)
class EntryRequest:
    """A vessel request to enter the port through a route to a berth."""
    call_id: str
    imo: str
    name: str
    loa_m: float
    beam_m: float
    draft_m: float
    route_id: str
    berth_id: str
    earliest_entry: datetime
    latest_entry: datetime
    tug_requirement: TugRequirement
    squat_allowance_m: float = 0.0
    wave_allowance_m: float = 0.0
    minimum_depth_margin_m: float = 0.5
    beam_clearance_factor: float = 1.5
    requested_speed_kn: float = 10.0
    pilot_district: str = ''
    berth_service: timedelta = timedelta(hours=4)
    # Tugs are released once the berthing manoeuvre is over; they do not wait
    # out cargo operations.
    tug_hold: timedelta = timedelta(hours=1)
    preferred_position_m: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'earliest_entry', ensure_utc(self.earliest_entry))
        object.__setattr__(self, 'latest_entry', ensure_utc(self.latest_entry))
        if not (self.imo.isdigit() and len(self.imo) == 7):
            raise ValidationError('IMO number must contain seven digits')
        if self.latest_entry <= self.earliest_entry:
            raise ValidationError('latest entry must follow earliest entry')
        if min(self.loa_m, self.beam_m, self.draft_m, self.requested_speed_kn) <= 0:
            raise ValidationError('vessel dimensions and speed must be positive')
        if self.minimum_depth_margin_m < 0 or self.beam_clearance_factor <= 0:
            raise ValidationError('invalid channel margins')
        if self.berth_service.total_seconds() <= 0:
            raise ValidationError('berth service duration must be positive')
        if self.tug_hold.total_seconds() <= 0:
            raise ValidationError('tug hold duration must be positive')


@dataclass(frozen=True)
class EntrySlot:
    """A clearance decision for one candidate entry time."""
    entry_at: datetime
    complete_at: datetime
    pilot_id: str
    tug_ids: tuple[str, ...]
    berth_id: str

    @property
    def interval(self) -> TimeInterval:
        return TimeInterval(self.entry_at, self.complete_at)


@dataclass
class EntryPlan:
    request: EntryRequest
    revision: int
    state: PlanState = PlanState.PROPOSED
    slot: EntrySlot | None = None
    last_slot: EntrySlot | None = None
    last_outcome: str | None = None
    history: list['PlanEvent'] = field(default_factory=list)

    @property
    def call_id(self) -> str:
        return self.request.call_id

    @property
    def scheduled(self) -> bool:
        return self.slot is not None

    def to_dict(self) -> dict[str, object]:
        return {
            'call_id': self.call_id,
            'vessel': self.request.name,
            'imo': self.request.imo,
            'state': self.state.value,
            'revision': self.revision,
            'outcome': self.last_outcome,
            'slot': None
            if self.slot is None
            else {
                'entry_at': iso_z(self.slot.entry_at),
                'complete_at': iso_z(self.slot.complete_at),
                'pilot_id': self.slot.pilot_id,
                'tug_ids': list(self.slot.tug_ids),
                'berth_id': self.slot.berth_id,
            },
            'history': [e.to_dict() for e in self.history],
        }


@dataclass(frozen=True)
class DenyReason:
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {'code': self.code, 'detail': self.detail}


@dataclass(frozen=True)
class PlanEvent:
    """One rejection, postponement, scheduling or lifecycle action."""
    at: datetime
    kind: str
    outcome: str
    detail: str
    reasons: tuple[DenyReason, ...] = ()
    slot: EntrySlot | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            'at': iso_z(self.at),
            'kind': self.kind,
            'outcome': self.outcome,
            'detail': self.detail,
            'reasons': [r.to_dict() for r in self.reasons],
            'slot': None
            if self.slot is None
            else {
                'entry_at': iso_z(self.slot.entry_at),
                'complete_at': iso_z(self.slot.complete_at),
                'pilot_id': self.slot.pilot_id,
                'tug_ids': list(self.slot.tug_ids),
                'berth_id': self.slot.berth_id,
            },
        }


@dataclass(frozen=True)
class PlanningDecision:
    call_id: str
    outcome: str  # scheduled | blocked | unchanged
    slot: EntrySlot | None
    reasons: tuple[DenyReason, ...]
    revision: int

    @property
    def approved(self) -> bool:
        return self.slot is not None

    def to_dict(self) -> dict[str, object]:
        return {
            'call_id': self.call_id,
            'outcome': self.outcome,
            'approved': self.approved,
            'revision': self.revision,
            'slot': None if self.slot is None else {
                'entry_at': iso_z(self.slot.entry_at),
                'complete_at': iso_z(self.slot.complete_at),
                'pilot_id': self.slot.pilot_id,
                'tug_ids': list(self.slot.tug_ids),
                'berth_id': self.slot.berth_id,
            },
            'reasons': [r.to_dict() for r in self.reasons],
        }


class PortEntryPlanner:
    """Chains the six independent clearances into queryable entry plans."""

    def __init__(
        self,
        weather_limits: MarineWeatherLimits,
        step: timedelta = timedelta(minutes=15),
        weather_max_gap: timedelta = timedelta(hours=1),
        night_window: tuple[float, float] | None = None,
    ):
        if step.total_seconds() <= 0 or weather_max_gap.total_seconds() <= 0:
            raise ValidationError('step and weather gap must be positive')
        if night_window is not None:
            start_h, end_h = night_window
            if start_h == end_h or not 0.0 <= start_h < 24.0 or not 0.0 < end_h <= 24.0:
                raise ValidationError('night window hours must stay inside 0..24')
        self.weather_checker = WeatherWindowChecker(weather_limits)
        self.step = step
        self.weather_max_gap = weather_max_gap
        self.night_window = night_window
        self.tide_curves: dict[str, TideCurve] = {}
        self.weather: list[WeatherSample] = []
        self.routes: dict[str, ChannelRoute] = {}
        self.pilots: list[Pilot] = []
        self.tugs: list[Tug] = []
        self.berths: dict[str, Berth] = {}
        self.closures = BerthWindowCalendar()
        self.plans: dict[str, EntryPlan] = {}

    # ----- forecast / resource ingestion ----------------------------------

    def set_tide_curve(self, leg_id: str, curve: TideCurve) -> None:
        self.tide_curves[leg_id] = curve

    def update_weather(self, samples: list[WeatherSample]) -> None:
        items = sorted(samples, key=lambda s: s.at)
        if any(a.at == b.at for a, b in zip(items, items[1:])):
            raise ValidationError('duplicate weather sample times are not allowed')
        self.weather = items

    def set_route(self, route_id: str, route: ChannelRoute) -> None:
        self.routes[route_id] = route

    def set_pilots(self, pilots: list[Pilot]) -> None:
        ids = [p.pilot_id for p in pilots]
        if len(set(ids)) != len(ids):
            raise ValidationError('duplicate pilot id')
        self.pilots = list(pilots)

    def set_tugs(self, tugs: list[Tug]) -> None:
        ids = [t.tug_id for t in tugs]
        if len(set(ids)) != len(ids):
            raise ValidationError('duplicate tug id')
        self.tugs = list(tugs)

    def add_berth(self, berth: Berth) -> None:
        self.berths[berth.berth_id] = berth

    def add_closure(self, closure: BerthClosure) -> None:
        self.closures.add_closure(closure)

    # ----- queries --------------------------------------------------------

    def get_plan(self, call_id: str) -> EntryPlan:
        try:
            return self.plans[call_id]
        except KeyError as exc:
            raise NotFoundError(f'no entry plan for call {call_id}') from exc

    def list_plans(self, state: PlanState | None = None) -> tuple[EntryPlan, ...]:
        items = self.plans.values()
        if state is not None:
            items = (p for p in items if p.state == state)
        return tuple(sorted(items, key=lambda p: p.call_id))

    def pending_rejections(self) -> tuple[EntryPlan, ...]:
        return tuple(
            sorted(
                (p for p in self.plans.values() if p.state == PlanState.PROPOSED and p.slot is None),
                key=lambda p: p.call_id,
            )
        )

    # ----- plan lifecycle -------------------------------------------------

    def submit(self, request: EntryRequest, at: datetime | None = None) -> PlanningDecision:
        at = ensure_utc(at) if at else request.earliest_entry
        if request.call_id in self.plans:
            raise ConflictError(f'plan for call {request.call_id} already exists')
        plan = EntryPlan(request=request, revision=1)
        self.plans[request.call_id] = plan
        slot, reasons = self._search(request, self._frozen())
        decision = self._apply_run(plan, slot, reasons, at, 'submitted')
        return decision

    def confirm(self, call_id: str, at: datetime) -> PlanningDecision:
        at = ensure_utc(at)
        plan = self._get(call_id)
        if plan.state == PlanState.EXECUTED:
            raise StateTransitionError('executed plan cannot be confirmed')
        if plan.state == PlanState.CONFIRMED:
            raise StateTransitionError('plan is already confirmed')
        if plan.slot is None:
            raise StateTransitionError('cannot confirm a blocked plan')
        slot, reasons = self._evaluate(
            plan.request, plan.slot.entry_at, self._frozen(exclude={call_id})
        )
        if slot is None:
            plan.last_slot = plan.slot
            plan.slot = None
            plan.last_outcome = 'blocked'
            plan.revision += 1
            event = PlanEvent(at, 'confirm_failed', 'blocked',
                              'clearance no longer valid at confirmation', reasons)
            plan.history.append(event)
            return PlanningDecision(call_id, 'blocked', None, tuple(reasons), plan.revision)
        plan.state = PlanState.CONFIRMED
        plan.last_outcome = 'confirmed'
        plan.history.append(PlanEvent(at, 'confirmed', 'confirmed',
                                      'entry plan confirmed and frozen', (), slot))
        return PlanningDecision(call_id, 'unchanged', slot, (), plan.revision)

    def execute(self, call_id: str, at: datetime) -> PlanningDecision:
        at = ensure_utc(at)
        plan = self._get(call_id)
        if plan.state == PlanState.EXECUTED:
            raise StateTransitionError('plan is already executed')
        if plan.state != PlanState.CONFIRMED:
            raise StateTransitionError('only a confirmed plan can be executed')
        plan.state = PlanState.EXECUTED
        plan.last_outcome = 'executed'
        plan.history.append(PlanEvent(at, 'executed', 'executed',
                                      'entry recorded as executed', (), plan.slot))
        return PlanningDecision(call_id, 'unchanged', plan.slot, (), plan.revision)

    def replan_unconfirmed(self, at: datetime) -> list[PlanningDecision]:
        """Replan every proposed plan after a forecast or resource change.

        Confirmed and executed records never move. Rechecking and resource
        allocation process proposals in a deterministic order so that earlier
        calls keep priority for contested pilots, tugs and berth space.
        """
        at = ensure_utc(at)
        decisions: list[PlanningDecision] = []
        proposals = sorted(
            (p for p in self.plans.values() if p.state == PlanState.PROPOSED),
            key=lambda p: (p.request.earliest_entry, p.call_id),
        )
        for plan in proposals:
            old = plan.slot
            frozen = self._frozen(exclude={plan.call_id})
            if old is not None:
                slot, reasons = self._evaluate(plan.request, old.entry_at, frozen)
                if slot is not None:
                    plan.last_outcome = 'unchanged'
                    plan.history.append(PlanEvent(
                        at, 'replan_checked', 'unchanged',
                        'current slot still satisfies every clearance', (), slot))
                    decisions.append(PlanningDecision(plan.call_id, 'unchanged', slot, (), plan.revision))
                    continue
            else:
                reasons = []
            slot, search_reasons = self._search(plan.request, frozen)
            if slot is not None:
                detail = ('new feasible slot found after change' if old is None
                          else f'postponed from {iso_z(old.entry_at)}')
                outcome = 'scheduled' if old is None else 'postponed'
                blocked_reasons = tuple(reasons)
                plan.last_slot = old
                plan.slot = slot
                plan.last_outcome = outcome
                plan.revision += 1
                plan.history.append(PlanEvent(
                    at, 'replanned', outcome, detail, blocked_reasons, slot))
                decisions.append(PlanningDecision(plan.call_id, outcome, slot, blocked_reasons, plan.revision))
            else:
                all_reasons = self._merge_reasons(reasons + search_reasons)
                plan.last_slot = old
                plan.slot = None
                plan.last_outcome = 'blocked'
                plan.revision += 1
                detail = ('no feasible slot after change' if old is None
                          else f'previously scheduled at {iso_z(old.entry_at)}; no feasible slot after change')
                plan.history.append(PlanEvent(
                    at, 'replan_blocked', 'blocked', detail, tuple(all_reasons)))
                decisions.append(PlanningDecision(
                    plan.call_id, 'blocked', None, tuple(all_reasons), plan.revision))
        return decisions

    def explain(self, request: EntryRequest, entry_at: datetime) -> PlanningDecision:
        """Dry-run a single candidate time without creating or changing plans."""
        entry_at = ensure_utc(entry_at)
        slot, reasons = self._evaluate(request, entry_at, self._frozen())
        return PlanningDecision(
            request.call_id,
            'scheduled' if slot is not None else 'blocked',
            slot,
            tuple(reasons),
            self.plans.get(request.call_id).revision if request.call_id in self.plans else 0,
        )

    def why_blocked(self, call_id: str) -> tuple[DenyReason, ...]:
        plan = self._get(call_id)
        for event in reversed(plan.history):
            if event.outcome == 'blocked':
                return event.reasons
        return ()

    # ----- internals ------------------------------------------------------

    def _get(self, call_id: str) -> EntryPlan:
        try:
            return self.plans[call_id]
        except KeyError as exc:
            raise NotFoundError(f'no entry plan for call {call_id}') from exc

    def _frozen(self, exclude: set[str] | None = None) -> tuple[EntryPlan, ...]:
        exclude = exclude or set()
        return tuple(
            p for p in self.plans.values()
            if p.call_id not in exclude and p.slot is not None
            and p.state in {PlanState.PROPOSED, PlanState.CONFIRMED, PlanState.EXECUTED}
        )

    def _apply_run(self, plan: EntryPlan, slot: EntrySlot | None,
                   reasons: list[DenyReason], at: datetime, kind: str) -> PlanningDecision:
        req = plan.request
        if slot is not None:
            plan.slot = slot
            plan.last_outcome = 'scheduled'
            plan.history.append(PlanEvent(
                at, kind, 'scheduled',
                f'earliest feasible entry at {iso_z(slot.entry_at)}', (), slot))
            return PlanningDecision(req.call_id, 'scheduled', slot, (), plan.revision)
        plan.slot = None
        plan.last_outcome = 'blocked'
        plan.history.append(PlanEvent(
            at, kind + '_blocked', 'blocked',
            'no feasible slot inside the requested window', tuple(reasons)))
        return PlanningDecision(req.call_id, 'blocked', None, tuple(reasons), plan.revision)

    def _search(self, req: EntryRequest, frozen: tuple[EntryPlan, ...]
                ) -> tuple[EntrySlot | None, list[DenyReason]]:
        step = self.step
        cursor = req.earliest_entry
        collected: list[DenyReason] = []
        while cursor <= req.latest_entry:
            slot, reasons = self._evaluate(req, cursor, frozen)
            if slot is not None:
                return slot, []
            collected = self._merge_reasons(collected + reasons)
            cursor += step
        return None, collected

    @staticmethod
    def _merge_reasons(reasons: list[DenyReason]) -> list[DenyReason]:
        seen: dict[str, str] = {}
        order: list[str] = []
        for reason in reasons:
            if reason.code not in seen:
                seen[reason.code] = reason.detail
                order.append(reason.code)
        return [DenyReason(code, seen[code]) for code in order]

    def _leg_offsets(self, route: ChannelRoute, req: EntryRequest
                     ) -> list[tuple[str, float, float, float]]:
        """Return (leg_id, entry_offset_h, hours, chart_depth_m) per leg."""
        offsets = []
        elapsed = 0.0
        for leg in route.legs:
            speed = min(req.requested_speed_kn, leg.speed_limit_kn)
            hours = leg.length_nm / speed
            offsets.append((leg.leg_id, elapsed, hours, leg.chart_depth_m))
            elapsed += hours
        return offsets

    def _is_night(self, interval: TimeInterval) -> bool:
        if self.night_window is None:
            return False
        start_minutes = self.night_window[0] * 60
        end_minutes = self.night_window[1] * 60
        cursor = interval.start
        while cursor < interval.end:
            minutes = cursor.hour * 60 + cursor.minute
            if start_minutes <= end_minutes:
                night = start_minutes <= minutes < end_minutes
            else:  # window crosses midnight
                night = minutes >= start_minutes or minutes < end_minutes
            if night:
                return True
            cursor = min(cursor + timedelta(minutes=5), interval.end)
        return False

    def _evaluate(self, req: EntryRequest, entry_at: datetime,
                  frozen: tuple[EntryPlan, ...]) -> tuple[EntrySlot | None, list[DenyReason]]:
        reasons: list[DenyReason] = []
        route = self.routes.get(req.route_id)
        if route is None:
            reasons.append(DenyReason(CHANNEL_ROUTE_MISSING,
                                      f'route {req.route_id} is not defined'))
        if not self.weather:
            reasons.append(DenyReason(WEATHER_MISSING, 'no weather forecast loaded'))
        if not self.pilots:
            reasons.append(DenyReason(PILOT_ROSTER_MISSING, 'no pilot roster loaded'))
        if not self.tugs:
            reasons.append(DenyReason(TUG_FLEET_MISSING, 'no tug fleet loaded'))
        berth = self.berths.get(req.berth_id)
        if berth is None:
            reasons.append(DenyReason(BERTH_MISSING, f'berth {req.berth_id} is not defined'))
        # Tide curves are required per channel leg and fail closed.
        tide_heights: dict[str, float] = {}
        transit_hours = 0.0
        if route is not None:
            offsets = self._leg_offsets(route, req)
            for leg_id, offset, hours, chart_depth in offsets:
                curve = self.tide_curves.get(leg_id)
                if curve is None:
                    reasons.append(DenyReason(
                        TIDE_MISSING, f'no tide forecast covering channel leg {leg_id}'))
                    continue
                moment = entry_at + timedelta(hours=offset + hours / 2)
                try:
                    height = curve.height_at(moment)
                except SafetyConstraintError:
                    reasons.append(DenyReason(
                        TIDE_MISSING,
                        f'tide forecast for leg {leg_id} does not cover {iso_z(moment)}'))
                    continue
                tide_heights[leg_id] = height
                ukc = UnderKeelRequirement(
                    chart_depth, req.draft_m,
                    max(0.0, req.minimum_depth_margin_m),
                    req.squat_allowance_m, req.wave_allowance_m)
                if curve.clearance_m(moment, ukc) < ukc.minimum_clearance_m:
                    reasons.append(DenyReason(
                        TIDE_UKC,
                        f'leg {leg_id} under-keel clearance insufficient at {iso_z(moment)}'))
            transit_hours = sum(h for _, _, h, _ in offsets)

        if route is not None and not any(r.code == TIDE_MISSING for r in reasons):
            transit_req = TransitRequirement(
                req.draft_m, req.beam_m, req.requested_speed_kn,
                req.minimum_depth_margin_m, req.beam_clearance_factor)
            try:
                route.evaluate(transit_req, tide_heights)
            except SafetyConstraintError as exc:
                message = str(exc)
                code = CHANNEL_WIDTH if 'width' in message else CHANNEL_DEPTH
                reasons.append(DenyReason(code, message))
        if route is None:
            # Structural data is already reported; the other checks need a route.
            return None, self._merge_reasons(reasons)

        complete_at = entry_at + timedelta(hours=transit_hours) + req.berth_service
        transit_interval = TimeInterval(entry_at, entry_at + timedelta(hours=transit_hours))

        # Weather must cover the whole transit and stay inside limits.
        if self.weather:
            times = [s.at for s in self.weather]
            first = bisect_right(times, transit_interval.start) - 1
            covering = self.weather[max(0, first):]
            covering = [s for s in covering if s.at <= transit_interval.end]
            # The last observation remains usable for one forecast cadence;
            # beyond that the forecast is treated as absent (fail closed).
            covered_end = (covering[-1].at + self.weather_max_gap) if covering else None
            if not covering or covering[0].at > transit_interval.start \
                    or covered_end < transit_interval.end:
                reasons.append(DenyReason(
                    WEATHER_MISSING,
                    f'weather forecast does not cover transit {iso_z(transit_interval.start)}'
                    f'..{iso_z(transit_interval.end)}'))
            else:
                # Gaps fully inside the transit must stay within the cadence.
                bounded = [s for s in covering if s.at < transit_interval.end]
                gap_ok = all(
                    b.at - a.at <= self.weather_max_gap
                    for a, b in zip(bounded, bounded[1:])
                )
                if not gap_ok:
                    reasons.append(DenyReason(
                        WEATHER_GAP,
                        f'weather samples are more than {int(self.weather_max_gap.total_seconds() // 60)} '
                        'minutes apart during transit'))
                for sample in covering:
                    for token in self.weather_checker.reasons(sample):
                        reasons.append(DenyReason(
                            _WEATHER_CODE[token],
                            f'{token} limit exceeded at {iso_z(sample.at)}'))

        # Pilot: roster seeded with every already-allocated plan stays frozen.
        # Pilots only ride the transit, so frozen jobs occupy transit intervals.
        chosen_pilot: Pilot | None = None
        if self.pilots:
            roster = PilotRoster(self.pilots)
            for plan in frozen:
                frozen_transit = TimeInterval(
                    plan.slot.entry_at,
                    plan.slot.entry_at + timedelta(hours=self._transit_hours(plan.request)))
                roster.assignments.append(
                    PilotAssignment(plan.slot.pilot_id, plan.call_id, frozen_transit))
            district = req.pilot_district or route.legs[0].leg_id
            job = PilotJob(req.call_id, district, req.loa_m, transit_interval,
                           self._is_night(transit_interval))
            try:
                chosen_pilot = roster.assign(job)
            except CapacityError:
                reasons.append(DenyReason(
                    PILOT_UNAVAILABLE,
                    'no qualified rested pilot available for the transit window'))

        # Tugs are held only for the berthing manoeuvre, not cargo work.
        tug_window = TimeInterval(entry_at, entry_at + req.tug_hold)
        chosen_tugs: tuple[Tug, ...] | None = None
        if self.tugs:
            allocator = TugAllocator(self.tugs)
            for plan in frozen:
                frozen_tug_window = TimeInterval(plan.slot.entry_at,
                                                 plan.slot.entry_at + plan.request.tug_hold)
                if frozen_tug_window.overlaps(tug_window):
                    allocator.busy.update(plan.slot.tug_ids)
            try:
                chosen_tugs = allocator.allocate(req.tug_requirement)
            except CapacityError:
                reasons.append(DenyReason(
                    TUG_UNAVAILABLE,
                    'tug fleet cannot meet pull, count, azimuthing or firefighting requirement'))

        # Berth must exist, accept the draft, stay open and be conflict free.
        if berth is not None:
            if req.draft_m > berth.max_draft_m:
                reasons.append(DenyReason(
                    BERTH_DRAFT,
                    f'vessel draft {req.draft_m} m exceeds berth {req.berth_id} limit {berth.max_draft_m} m'))
            occupancy = TimeInterval(entry_at + timedelta(hours=transit_hours), complete_at)
            frees = self.closures.available(req.berth_id, occupancy)
            if not any(f.start <= occupancy.start and f.end >= occupancy.end for f in frees):
                reasons.append(DenyReason(
                    BERTH_CLOSED,
                    f'berth {req.berth_id} is closed for {iso_z(occupancy.start)}..{iso_z(occupancy.end)}'))
            planner = BerthPlanner(berth)
            for plan in sorted(frozen, key=lambda p: (p.slot.entry_at, p.call_id)):
                if plan.slot.berth_id != req.berth_id:
                    continue
                other = plan.request
                planner.assign(VesselCall(
                    plan.call_id, other.loa_m, other.beam_m,
                    plan.slot.entry_at + timedelta(hours=self._transit_hours(other)),
                    plan.slot.complete_at, other.draft_m, other.preferred_position_m))
            try:
                planner.assign(VesselCall(
                    req.call_id, req.loa_m, req.beam_m, occupancy.start, occupancy.end,
                    req.draft_m, req.preferred_position_m))
            except ConflictError as exc:
                message = str(exc)
                code = BERTH_DRAFT if 'draft' in message else BERTH_OCCUPIED
                reasons.append(DenyReason(code, message))

        if reasons:
            return None, self._merge_reasons(reasons)
        return EntrySlot(
            entry_at, complete_at, chosen_pilot.pilot_id,
            tuple(t.tug_id for t in chosen_tugs), req.berth_id), []

    def _transit_hours(self, req: EntryRequest) -> float:
        route = self.routes.get(req.route_id)
        if route is None:
            return 0.0
        return sum(h for _, _, h, _ in self._leg_offsets(route, req))
