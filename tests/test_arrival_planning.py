from datetime import datetime, timedelta, timezone

import pytest

from terminalweave.core.errors import (
    ConflictError,
    NotFoundError,
    StateTransitionError,
    ValidationError,
)
from terminalweave.core.intervals import TimeInterval
from terminalweave.marine.arrival import (
    ArrivalPlanner,
    ArrivalPolicy,
    ArrivalRequest,
    ArrivalState,
    Gate,
    MarineForecast,
)
from terminalweave.marine.berth import Berth, BerthPlanner
from terminalweave.marine.berth_windows import BerthClosure, BerthWindowCalendar
from terminalweave.marine.channel import ChannelLeg, ChannelRoute
from terminalweave.marine.pilotage import Pilot, PilotRoster
from terminalweave.marine.tide import TideCurve, TideSample
from terminalweave.marine.tugs import Tug, TugAllocator, TugRequirement
from terminalweave.marine.weather import MarineWeatherLimits, WeatherSample

UTC = timezone.utc
D = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)


def at(hours):
    return D + timedelta(hours=hours)


def weather(start=None, hours=14, wind=15.0):
    start = start if start is not None else at(0)
    return tuple(
        WeatherSample(start + timedelta(hours=h), wind, wind + 5, 3.0, 1.0)
        for h in range(hours + 1)
    )


ROUTE = ChannelRoute(
    [ChannelLeg('SEA', 10.0, 200.0, 12.0, 10.0),
     ChannelLeg('BAR', 10.0, 160.0, 10.0, 10.0, one_way=True)]
)
LIMITS = MarineWeatherLimits(25.0, 35.0, 1.0, 2.0)
TUG_REQ = TugRequirement(70.0, 2, require_azimuthing=1)
POLICY = ArrivalPolicy(search_step=timedelta(minutes=30), max_transit_hours=4.0)


def tide(level=3.0):
    return TideCurve([TideSample(at(-1), level), TideSample(at(14), level)])


def forecast(rev='F1', *, tide_curve='__default__', samples=None, channel=ROUTE,
             limits=LIMITS, published_at=None, minimum_depth_margin_m=0.5):
    return MarineForecast(
        tide() if tide_curve == '__default__' else tide_curve,
        samples if samples is not None else weather(),
        limits,
        channel,
        rev,
        published_at or at(0),
        minimum_depth_margin_m=minimum_depth_margin_m,
    )


def make_planner(fc=None, pilots=None, tugs=None, berths=None, closures=None):
    return ArrivalPlanner(
        fc or forecast(),
        pilots or PilotRoster([Pilot('P1', 300.0, True, 'RIVER')]),
        tugs or TugAllocator([Tug('T1', 50.0, True), Tug('T2', 50.0, True)]),
        berths or {'B1': BerthPlanner(Berth('B1', 400.0, 14.0))},
        closures or BerthWindowCalendar(),
        POLICY,
    )


def request(call_id='V1', draft=11.0, loa=240.0, beam=32.0, earliest=4, latest=10,
            preferred='B1', stay_hours=2):
    return ArrivalRequest(
        call_id, loa, beam, draft, 'RIVER', TUG_REQ,
        at(earliest), at(latest), preferred_berth_id=preferred,
        berth_stay=timedelta(hours=stay_hours),
    )


def codes(plan, gate=None):
    return sorted({f.code for f in plan.findings if gate is None or f.gate == gate})


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------


def test_all_gates_pass_releases_first_window():
    planner = make_planner()
    plan = planner.submit(request())
    assert plan.state == ArrivalState.PROPOSED
    assert plan.window == TimeInterval(at(4), at(6))
    assert plan.transit_hours == 2.0
    assert plan.resources is not None
    assert plan.resources.pilot_id == 'P1'
    assert plan.resources.berth_id == 'B1'
    assert set(plan.resources.tug_ids) == {'T1', 'T2'}


def test_plan_is_queryable():
    planner = make_planner()
    plan = planner.submit(request())
    assert planner.get_plan('V1') is plan
    assert planner.plan_for_call('V1') is plan
    assert planner.plans(ArrivalState.PROPOSED) == (plan,)
    assert [p.plan_id for p in planner.released_plans()] == ['V1']
    payload = plan.to_dict()
    assert payload['state'] == 'proposed'
    assert payload['window']['start'].endswith('04:00:00Z')


def test_unknown_plan_raises():
    planner = make_planner()
    with pytest.raises(NotFoundError):
        planner.get_plan('NOPE')


def test_duplicate_call_rejected():
    planner = make_planner()
    planner.submit(request())
    with pytest.raises(ConflictError):
        planner.submit(request())


# ---------------------------------------------------------------------------
# Missing information fails closed
# ---------------------------------------------------------------------------


def test_missing_tide_forecast_fails_closed():
    fc = forecast(tide_curve=None)
    planner = make_planner(fc)
    plan = planner.submit(request())
    assert plan.state == ArrivalState.REJECTED
    assert codes(plan, Gate.TIDE.value) == ['forecast_missing']


def test_tide_forecast_not_covering_horizon_fails_closed():
    fc = forecast(tide_curve=TideCurve([TideSample(at(4), 3.0), TideSample(at(8), 3.0)]))
    planner = make_planner(fc)
    plan = planner.submit(request())
    assert codes(plan, Gate.TIDE.value) == ['forecast_gap']


def test_missing_weather_forecast_fails_closed():
    fc = forecast(samples=(), limits=None)
    planner = make_planner(fc)
    plan = planner.submit(request())
    assert codes(plan, Gate.WEATHER.value) == ['forecast_missing']


def test_weather_sample_gap_fails_closed():
    sparse = (
        WeatherSample(at(4), 15, 20, 3.0, 1.0),
        WeatherSample(at(8), 15, 20, 3.0, 1.0),
        WeatherSample(at(13), 15, 20, 3.0, 1.0),
    )
    planner = make_planner(forecast(samples=sparse))
    plan = planner.submit(request())
    assert codes(plan, Gate.WEATHER.value) == ['forecast_gap']


def test_missing_route_and_roster_all_reported():
    fc = forecast(channel=None)
    planner = make_planner(
        fc,
        pilots=PilotRoster([Pilot('P1', 300.0, True, 'OTHER')]),
        tugs=TugAllocator([]),
    )
    plan = planner.submit(request())
    assert plan.state == ArrivalState.REJECTED
    assert codes(plan, Gate.CHANNEL.value) == ['route_missing']
    assert codes(plan, Gate.PILOT.value) == ['district_missing']
    assert codes(plan, Gate.TUG.value) == ['fleet_missing']


# ---------------------------------------------------------------------------
# Individual safety gates
# ---------------------------------------------------------------------------


def test_insufficient_ukc_rejects_with_reason():
    planner = make_planner(forecast(tide_curve=tide(level=1.0)))
    plan = planner.submit(request(draft=11.0))
    assert codes(plan, Gate.TIDE.value) == ['insufficient_ukc']


def test_channel_depth_rejects():
    planner = make_planner(forecast(tide_curve=tide(level=0.0),
                                    minimum_depth_margin_m=0.5))
    plan = planner.submit(request(draft=11.0))
    assert codes(plan, Gate.CHANNEL.value) == ['insufficient_depth']


def test_wind_exceeds_limit_rejects():
    planner = make_planner(forecast(samples=weather(wind=40.0)))
    plan = planner.submit(request())
    assert codes(plan, Gate.WEATHER.value) == ['conditions_exceed_limits']
    finding = next(f for f in plan.findings if f.gate == Gate.WEATHER.value)
    assert 'sustained wind' in finding.detail


def test_no_night_qualified_pilot_rejects():
    planner = make_planner(
        pilots=PilotRoster([Pilot('P1', 300.0, False, 'RIVER')]))
    plan = planner.submit(request(earliest=2, latest=4))
    assert codes(plan, Gate.PILOT.value) == ['no_qualified_pilot']
    finding = next(f for f in plan.findings if f.gate == Gate.PILOT.value)
    assert 'night_qualification' in finding.detail


def test_pilot_loa_limit_rejects():
    planner = make_planner(
        pilots=PilotRoster([Pilot('P1', 200.0, True, 'RIVER')]))
    plan = planner.submit(request(loa=240.0))
    assert codes(plan, Gate.PILOT.value) == ['no_qualified_pilot']


def test_tug_capacity_rejects():
    planner = make_planner(
        tugs=TugAllocator([Tug('T1', 30.0, True), Tug('T2', 30.0, True)]))
    plan = planner.submit(request())
    assert codes(plan, Gate.TUG.value) == ['requirement_unmet']


def test_berth_draft_limit_rejects():
    planner = make_planner(
        berths={'B1': BerthPlanner(Berth('B1', 400.0, 10.0))})
    plan = planner.submit(request(draft=11.0))
    assert codes(plan, Gate.BERTH.value) == ['no_berth']
    finding = next(f for f in plan.findings if f.gate == Gate.BERTH.value)
    assert 'draft_limit' in finding.detail


def test_berth_closure_rejects():
    closures = BerthWindowCalendar()
    closures.add_closure(
        BerthClosure('B1', TimeInterval(at(0), at(12)), 'maintenance'))
    planner = make_planner(closures=closures)
    plan = planner.submit(request())
    finding = next(f for f in plan.findings if f.gate == Gate.BERTH.value)
    assert 'closure' in finding.detail


def test_confirmed_call_blocks_draft_from_its_resources():
    planner = make_planner()
    planner.submit(request('V1', preferred='B1'))
    planner.confirm('V1')
    second = planner.submit(request('V2', preferred='B1'))
    assert second.state == ArrivalState.REJECTED
    assert codes(second, Gate.BERTH.value) == ['no_berth']


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_rejected_plan_cannot_be_confirmed():
    planner = make_planner(forecast(tide_curve=tide(level=0.0)))
    plan = planner.submit(request())
    with pytest.raises(StateTransitionError):
        planner.confirm(plan.plan_id)


def test_execute_requires_confirmation():
    planner = make_planner()
    planner.submit(request())
    with pytest.raises(StateTransitionError):
        planner.execute('V1', at(6))


def test_execute_releases_pilot_and_tugs_but_keeps_berth_record():
    planner = make_planner()
    planner.submit(request('V1', earliest=4))
    planner.confirm('V1')
    planner.execute('V1', at(6))
    plan = planner.get_plan('V1')
    assert plan.state == ArrivalState.EXECUTED
    assert planner.pilots.assignments == []
    assert planner.tugs.busy == set()
    assert len(planner.berths['B1'].assignments) == 1


# ---------------------------------------------------------------------------
# Forecast / resource revisions only touch unconfirmed plans
# ---------------------------------------------------------------------------


def test_forecast_revision_reschedules_only_draft_plan():
    planner = make_planner(
        pilots=PilotRoster([
            Pilot('P1', 300.0, True, 'RIVER'),
            Pilot('P2', 300.0, True, 'RIVER'),
        ]),
        tugs=TugAllocator([Tug(f'T{i}', 40.0, True) for i in range(1, 7)]),
        berths={
            'B1': BerthPlanner(Berth('B1', 400.0, 14.0)),
            'B2': BerthPlanner(Berth('B2', 400.0, 14.0)),
        },
    )
    fixed = planner.submit(request('FIXED', earliest=4, latest=10, preferred='B1'))
    planner.confirm('FIXED')
    draft = planner.submit(request('DRAFT', earliest=4, latest=10, preferred='B2'))
    # Pilot rest buffer after FIXED's 04:00-06:00 job pushes DRAFT to 06:30.
    assert draft.window.start == at(6) + POLICY.search_step

    # Tide stays below the draft's UKC until 08:00, then recovers.
    low = forecast(
        'F2',
        tide_curve=TideCurve([
            TideSample(at(3), 1.2), TideSample(at(8), 2.0),
            TideSample(at(9), 3.0), TideSample(at(14), 3.0),
        ]),
        published_at=at(1),
    )
    report = planner.update_forecast(low)

    by_call = {i.call_id: i for i in report.items}
    assert by_call['FIXED'].decision == 'confirmed'
    assert by_call['FIXED'].window.start == at(4)
    assert by_call['DRAFT'].decision == 'rescheduled'
    assert by_call['DRAFT'].previous_window.start == at(6) + POLICY.search_step
    assert by_call['DRAFT'].window.start == at(8)
    reschedule_reasons = by_call['DRAFT'].findings
    assert any(f.code == 'insufficient_ukc' for f in reschedule_reasons)

    fixed_plan = planner.get_plan('FIXED')
    draft_plan = planner.get_plan('DRAFT')
    assert fixed_plan.state == ArrivalState.CONFIRMED
    assert fixed_plan.forecast_revision == 'F1'
    assert draft_plan.forecast_revision == 'F2'
    assert draft_plan.state == ArrivalState.RESCHEDULED


def test_executed_plan_immutable_on_forecast_change():
    planner = make_planner()
    planner.submit(request('DONE', earliest=4, latest=10))
    planner.confirm('DONE')
    planner.execute('DONE', at(6))
    report = planner.update_forecast(forecast('F2', tide_curve=tide(0.0)))
    item = next(i for i in report.items if i.call_id == 'DONE')
    assert item.decision == 'executed'
    assert planner.get_plan('DONE').state == ArrivalState.EXECUTED
    assert planner.get_plan('DONE').window.start == at(4)


def test_unchanged_window_stays_proposed_without_findings():
    planner = make_planner()
    planner.submit(request('V1'))
    report = planner.update_forecast(forecast('F2'))
    item = report.items[0]
    assert item.decision == 'unchanged'
    assert item.findings == ()
    assert planner.get_plan('V1').state == ArrivalState.PROPOSED


def test_rejected_plan_re_evaluated_and_released_when_forecast_recovers():
    planner = make_planner(forecast(tide_curve=tide(level=0.0)))
    plan = planner.submit(request())
    assert plan.state == ArrivalState.REJECTED
    report = planner.update_forecast(forecast('F2'))
    item = report.items[0]
    assert item.decision == 'rescheduled'
    assert item.window is not None
    assert planner.get_plan('V1').state == ArrivalState.RESCHEDULED


def test_forecast_can_reject_previously_released_draft():
    planner = make_planner()
    planner.submit(request('V1'))
    report = planner.update_forecast(forecast('F2', tide_curve=tide(level=0.0)))
    item = report.items[0]
    assert item.decision == 'rejected'
    assert item.window is None
    assert item.previous_window.start == at(4)
    assert any(f.gate == Gate.CHANNEL.value or f.gate == Gate.TIDE.value
               for f in item.findings)
    assert planner.get_plan('V1').state == ArrivalState.REJECTED


def test_resource_change_only_replans_unconfirmed_and_frees_resources():
    planner = make_planner()
    early = planner.submit(request('EARLY', earliest=4, latest=10))
    later = planner.submit(request('LATER', earliest=4, latest=10))
    planner.confirm('EARLY')
    assert later.state == ArrivalState.REJECTED
    assert codes(later, Gate.TUG.value) == ['requirement_unmet']

    planner.execute('EARLY', at(6))
    report = planner.resource_changed()
    item = next(i for i in report.items if i.call_id == 'LATER')
    assert item.decision == 'rescheduled'
    # Execution released the pilot; its job ended at 06:00 so 06:00 is free.
    assert item.window.start == at(6)
    executed = next(i for i in report.items if i.call_id == 'EARLY')
    assert executed.decision == 'executed'


def test_same_revision_is_rejected():
    planner = make_planner()
    planner.submit(request())
    with pytest.raises(ValidationError):
        planner.update_forecast(forecast('F1'))
