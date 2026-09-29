from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from terminalweave.core.intervals import TimeInterval
from terminalweave.core.errors import StateTransitionError, NotFoundError
from terminalweave.marine.berth import Berth
from terminalweave.marine.berth_windows import BerthClosure
from terminalweave.marine.channel import ChannelLeg, ChannelRoute
from terminalweave.marine.entry_plan import (
    BERTH_CLOSED,
    BERTH_MISSING,
    CHANNEL_ROUTE_MISSING,
    PILOT_ROSTER_MISSING,
    PILOT_UNAVAILABLE,
    TIDE_MISSING,
    TUG_FLEET_MISSING,
    WEATHER_MISSING,
    WEATHER_WIND,
    EntryRequest,
    PlanState,
    PortEntryPlanner,
)
from terminalweave.marine.pilotage import Pilot
from terminalweave.marine.tide import TideCurve, TideSample
from terminalweave.marine.tugs import Tug, TugRequirement
from terminalweave.marine.weather import MarineWeatherLimits, WeatherSample

UTC = timezone.utc
D = datetime(2026, 1, 1, tzinfo=UTC)
IMO = {'C1': '1000001', 'C2': '1000002', 'C3': '1000003'}


def weather(wind=10.0, lightning=False, hours=24):
    return [
        WeatherSample(D + timedelta(minutes=30 * k), wind, 12, 5, 1.0, lightning)
        for k in range(hours * 2 + 1)
    ]


def make_planner(pilots=('P1',), tugs=(('T1', 60.0), ('T2', 60.0)), night=None):
    planner = PortEntryPlanner(
        MarineWeatherLimits(30.0, 40.0, 1.0, 3.0),
        step=timedelta(minutes=30),
        night_window=night,
    )
    planner.set_route('R1', ChannelRoute([ChannelLeg('HAR', 6.0, 200.0, 12.0, 12.0)]))
    planner.set_tide_curve(
        'HAR', TideCurve([TideSample(D, 1.0), TideSample(D + timedelta(hours=24), 1.0)]))
    planner.update_weather(weather())
    planner.set_pilots([Pilot(pid, 300.0, False, 'HAR') for pid in pilots])
    planner.set_tugs([Tug(tid, pull, True) for tid, pull in tugs])
    planner.add_berth(Berth('B1', 400.0, 16.0))
    return planner


def make_request(call_id='C1', earliest=D, latest=None, tugs=None, route_id='R1',
                 berth_id='B1', district='HAR'):
    return EntryRequest(
        call_id, IMO[call_id], f'MV {call_id}', 200.0, 32.0, 11.0,
        route_id, berth_id, earliest, latest or D + timedelta(hours=8),
        tugs or TugRequirement(80.0, 2),
        berth_service=timedelta(hours=2), pilot_district=district)


def test_chains_all_six_clearances_into_earliest_slot():
    planner = make_planner()
    decision = planner.submit(make_request())
    assert decision.outcome == 'scheduled'
    assert decision.approved
    assert decision.slot.entry_at == D
    assert decision.slot.pilot_id == 'P1'
    assert set(decision.slot.tug_ids) == {'T1', 'T2'}
    assert decision.slot.berth_id == 'B1'
    plan = planner.get_plan('C1')
    assert plan.state == PlanState.PROPOSED
    assert plan.history[0].kind == 'submitted'


def test_missing_any_dataset_fails_closed_with_reasons():
    planner = PortEntryPlanner(MarineWeatherLimits(30.0, 40.0, 1.0, 3.0))
    decision = planner.submit(make_request())
    assert decision.outcome == 'blocked'
    assert not decision.approved
    codes = {r.code for r in decision.reasons}
    assert {
        CHANNEL_ROUTE_MISSING,
        WEATHER_MISSING,
        PILOT_ROSTER_MISSING,
        TUG_FLEET_MISSING,
        BERTH_MISSING,
    } <= codes
    plan = planner.get_plan('C1')
    assert plan.slot is None
    assert {r.code for r in planner.why_blocked('C1')} == codes


def test_missing_tide_forecast_for_window_blocks_every_candidate():
    planner = make_planner()
    planner.set_tide_curve(
        'HAR',
        TideCurve([TideSample(D + timedelta(hours=10), 1.0),
                   TideSample(D + timedelta(hours=24), 1.0)]))
    decision = planner.submit(make_request())
    assert decision.outcome == 'blocked'
    assert {r.code for r in decision.reasons} == {TIDE_MISSING}


def test_weather_storm_postpones_unconfirmed_plan_and_records_reason():
    planner = make_planner()
    first = planner.submit(make_request())
    assert first.slot.entry_at == D
    # Storm covers the 00:00 transit but clears before the 01:00 candidate.
    planner.update_weather([
        WeatherSample(D + timedelta(minutes=30 * k),
                      40.0 if k <= 1 else 10.0, 12.0, 5.0, 1.0)
        for k in range(49)
    ])
    decisions = planner.replan_unconfirmed(D + timedelta(minutes=30))
    assert len(decisions) == 1
    moved = decisions[0]
    assert moved.outcome == 'postponed'
    assert moved.slot.entry_at == D + timedelta(hours=1)
    assert [r.code for r in moved.reasons] == [WEATHER_WIND]
    plan = planner.get_plan('C1')
    assert plan.last_slot.entry_at == D
    assert plan.revision == 2
    kinds = [(e.kind, e.outcome) for e in plan.history]
    assert ('submitted', 'scheduled') in kinds
    assert ('replanned', 'postponed') in kinds
    postponed = next(e for e in plan.history if e.outcome == 'postponed')
    assert postponed.reasons[0].code == WEATHER_WIND


def test_unchanged_forecast_keeps_slot_and_revision():
    planner = make_planner()
    planner.submit(make_request())
    decisions = planner.replan_unconfirmed(D + timedelta(hours=1))
    assert decisions[0].outcome == 'unchanged'
    assert decisions[0].slot.entry_at == D
    assert planner.get_plan('C1').revision == 1


def test_blocked_plan_is_scheduled_when_forecast_recovers():
    planner = make_planner()
    planner.update_weather(weather(wind=40.0))
    blocked = planner.submit(make_request())
    assert blocked.outcome == 'blocked'
    planner.update_weather(weather())
    decisions = planner.replan_unconfirmed(D + timedelta(hours=1))
    assert decisions[0].outcome == 'scheduled'
    assert decisions[0].slot.entry_at == D
    plan = planner.get_plan('C1')
    assert [e.outcome for e in plan.history] == ['blocked', 'scheduled']


def test_confirmed_plan_is_never_replanned():
    planner = make_planner()
    planner.submit(make_request())
    planner.confirm('C1', D + timedelta(minutes=5))
    assert planner.get_plan('C1').state == PlanState.CONFIRMED
    planner.update_weather(weather(wind=50.0))
    decisions = planner.replan_unconfirmed(D + timedelta(hours=1))
    assert decisions == []
    plan = planner.get_plan('C1')
    assert plan.state == PlanState.CONFIRMED
    assert plan.slot.entry_at == D


def test_confirmed_plan_survives_execution_and_executed_plans_are_frozen():
    planner = make_planner()
    planner.submit(make_request())
    planner.confirm('C1', D + timedelta(minutes=5))
    decision = planner.execute('C1', D + timedelta(minutes=10))
    assert decision.outcome == 'unchanged'
    plan = planner.get_plan('C1')
    assert plan.state == PlanState.EXECUTED
    planner.update_weather(weather(wind=50.0))
    assert planner.replan_unconfirmed(D + timedelta(hours=2)) == []
    assert plan.slot.entry_at == D
    with pytest.raises(StateTransitionError):
        planner.confirm('C1', D + timedelta(hours=3))


def test_only_proposed_plans_are_replanned_in_deterministic_order():
    planner = make_planner(pilots=('P1', 'P2'))
    planner.submit(make_request('C1'))
    planner.submit(make_request(
        'C2', earliest=D + timedelta(minutes=30),
        latest=D + timedelta(hours=8)))
    planner.confirm('C1', D + timedelta(minutes=5))
    planner.update_weather([
        WeatherSample(D + timedelta(minutes=30 * k), 10.0, 12.0, 5.0, 1.0,
                      lightning=(k == 1))
        for k in range(49)
    ])
    decisions = planner.replan_unconfirmed(D + timedelta(hours=1))
    assert [d.call_id for d in decisions] == ['C2']
    assert planner.get_plan('C1').slot.entry_at == D
    assert planner.get_plan('C1').state == PlanState.CONFIRMED


def test_single_pilot_serializes_two_entries_with_rest():
    planner = make_planner(pilots=('P1',))
    planner.submit(make_request('C1'))
    second = planner.submit(make_request(
        'C2', latest=D + timedelta(hours=12)))
    # Same pilot after 30 min rest and same berth after C1 clears: the first
    # feasible grid candidate is 02:00.
    assert second.slot.pilot_id == 'P1'
    assert second.slot.entry_at == D + timedelta(hours=2)
    history = planner.get_plan('C2').history
    assert history[0].outcome == 'scheduled'


def test_berth_closure_postpones_entry():
    planner = make_planner()
    planner.add_closure(BerthClosure(
        'B1', TimeInterval(D + timedelta(minutes=15), D + timedelta(hours=2)),
        'drill'))
    decision = planner.submit(make_request())
    assert decision.outcome == 'scheduled'
    # Berth occupancy starts 30 minutes after entry; the earliest placement
    # starts exactly when the closure lifts at 02:00.
    assert decision.slot.entry_at == D + timedelta(hours=1, minutes=30)


def test_window_fully_blocked_returns_berth_reason():
    planner = make_planner()
    planner.add_closure(BerthClosure(
        'B1', TimeInterval(D - timedelta(hours=1), D + timedelta(hours=10)),
        'drill'))
    decision = planner.submit(make_request())
    assert decision.outcome == 'blocked'
    assert [r.code for r in decision.reasons] == [BERTH_CLOSED]


def test_night_transit_requires_night_qualified_pilot():
    night = make_planner(night=(20.0, 6.0))
    evening = make_request(
        earliest=datetime(2026, 1, 1, 21, 0, tzinfo=UTC),
        latest=datetime(2026, 1, 1, 23, 0, tzinfo=UTC))
    blocked = night.submit(evening)
    assert blocked.outcome == 'blocked'
    assert [r.code for r in blocked.reasons] == [PILOT_UNAVAILABLE]
    qualified = make_planner(night=(20.0, 6.0))
    qualified.set_pilots([Pilot('PN', 300.0, True, 'HAR')])
    decision = qualified.submit(evening)
    assert decision.outcome == 'scheduled'
    assert decision.slot.entry_at == datetime(2026, 1, 1, 21, 0, tzinfo=UTC)


def test_explain_does_not_create_or_mutate_plans():
    planner = make_planner()
    decision = planner.explain(make_request(), D)
    assert decision.outcome == 'scheduled'
    with pytest.raises(NotFoundError):
        planner.get_plan('C1')
    blocked = planner.explain(make_request(berth_id='NOPE'), D)
    assert blocked.outcome == 'blocked'
    assert [r.code for r in blocked.reasons] == [BERTH_MISSING]


def test_confirmation_revalidates_and_keeps_rejection_history_on_failure():
    planner = make_planner()
    planner.submit(make_request())
    planner.update_weather(weather(wind=50.0))
    decision = planner.confirm('C1', D + timedelta(minutes=5))
    assert decision.outcome == 'blocked'
    plan = planner.get_plan('C1')
    assert plan.state == PlanState.PROPOSED
    assert plan.slot is None
    assert plan.last_slot is not None
    assert any(e.kind == 'confirm_failed' and e.outcome == 'blocked'
               for e in plan.history)


def test_plans_are_queryable_by_state_and_serializable():
    planner = make_planner(pilots=('P1', 'P2'))
    planner.submit(make_request('C1'))
    planner.submit(make_request('C2', earliest=D + timedelta(days=1),
                                latest=D + timedelta(days=2)))
    planner.confirm('C1', D + timedelta(minutes=5))
    assert [p.call_id for p in planner.list_plans(PlanState.PROPOSED)] == ['C2']
    assert [p.call_id for p in planner.list_plans(PlanState.CONFIRMED)] == ['C1']
    payload = planner.get_plan('C1').to_dict()
    assert payload['state'] == 'confirmed'
    assert payload['slot']['entry_at'] == '2026-01-01T00:00:00Z'
    assert payload['slot']['tug_ids'] == ['T1', 'T2']
    assert payload['history'][0]['at'].endswith('Z')


def test_duplicate_submit_and_unknown_call_are_rejected():
    planner = make_planner()
    planner.submit(make_request())
    with pytest.raises(Exception):
        planner.submit(make_request())
    with pytest.raises(NotFoundError):
            planner.get_plan('nope')
