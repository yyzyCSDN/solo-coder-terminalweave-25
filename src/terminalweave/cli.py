from __future__ import annotations
import argparse, json
from datetime import datetime, timedelta, timezone
from terminalweave.marine.tide import TideCurve, TideSample, UnderKeelRequirement
from terminalweave.marine.channel import ChannelRoute, ChannelLeg
from terminalweave.marine.weather import WeatherSample, MarineWeatherLimits
from terminalweave.marine.pilotage import Pilot, PilotRoster
from terminalweave.marine.tugs import Tug, TugAllocator, TugRequirement
from terminalweave.marine.berth import Berth, BerthPlanner
from terminalweave.marine.berth_windows import BerthWindowCalendar
from terminalweave.marine.arrival import ArrivalPlanner, ArrivalRequest, ArrivalPolicy, MarineForecast
from terminalweave.api.http import serve
UTC = timezone.utc


def _weather(start, hours, wind=15):
    return tuple(
        WeatherSample(start + timedelta(hours=h), wind, wind + 5, 3.0, 1.0)
        for h in range(hours + 1)
    )


def arrival_demo() -> dict[str, object]:
    day = datetime(2026, 9, 29, 0, 0, tzinfo=UTC)

    def at(h):
        return day + timedelta(hours=h)

    route = ChannelRoute([
        ChannelLeg('SEA', 10.0, 200.0, 12.0, 10.0),
        ChannelLeg('BAR', 10.0, 160.0, 10.0, 10.0, one_way=True),
    ])
    limits = MarineWeatherLimits(25.0, 35.0, 1.0, 2.0)
    f1 = MarineForecast(
        TideCurve([TideSample(at(0), 2.5), TideSample(at(12), 3.0),
                   TideSample(at(24), 2.5), TideSample(at(36), 2.75)]),
        _weather(at(0), 48),
        limits,
        route,
        revision='F1',
        published_at=at(-24),
    )
    pilots = PilotRoster([Pilot('P1', 300.0, True, 'RIVER')])
    tugs = TugAllocator([Tug('T1', 45.0, True), Tug('T2', 45.0, True),
                         Tug('T3', 45.0, True), Tug('T4', 45.0, True)])
    berths = {
        'B1': BerthPlanner(Berth('B1', 400.0, 14.0)),
        'B2': BerthPlanner(Berth('B2', 400.0, 14.0)),
    }
    planner = ArrivalPlanner(
        f1, pilots, tugs, berths, BerthWindowCalendar(),
        ArrivalPolicy(search_step=timedelta(minutes=30)),
    )
    tug_req = TugRequirement(70.0, 2, require_azimuthing=1)
    plan_a = planner.submit(ArrivalRequest(
        'CALL-A', 240.0, 32.0, 11.0, 'RIVER', tug_req,
        at(8), at(18), preferred_berth_id='B2', berth_stay=timedelta(hours=6)))
    plan_b = planner.submit(ArrivalRequest(
        'CALL-B', 200.0, 30.0, 10.0, 'RIVER', tug_req,
        at(4), at(8), preferred_berth_id='B1', berth_stay=timedelta(hours=2)))
    plan_c = planner.submit(ArrivalRequest(
        'CALL-C', 250.0, 35.0, 11.5, 'RIVER', tug_req,
        at(8), at(20), preferred_berth_id='B1', berth_stay=timedelta(hours=4)))
    initial = {p.plan_id: p.to_dict() for p in planner.plans()}

    planner.confirm(plan_a.plan_id)
    planner.confirm(plan_b.plan_id)
    planner.execute(plan_b.plan_id, at(6))
    replan_after_execution = planner.resource_changed().to_dict()

    f2 = MarineForecast(
        TideCurve([TideSample(at(0), 1.5), TideSample(at(8), 2.0),
                   TideSample(at(10), 2.3), TideSample(at(11), 2.6),
                   TideSample(at(18), 3.0), TideSample(at(36), 2.5),
                   TideSample(at(48), 2.5)]),
        _weather(at(0), 48),
        limits,
        route,
        revision='F2',
        published_at=at(0),
    )
    replan_f2 = planner.update_forecast(f2).to_dict()
    return {
        'initial': initial,
        'after_resource_change': replan_after_execution,
        'after_forecast_f2': replan_f2,
        'final': {p.plan_id: {'state': p.state.value,
                              'window_start': p.window.start.isoformat() if p.window else None,
                              'revision': p.forecast_revision}
                  for p in planner.plans()},
    }


def demo() -> dict[str, object]:
    start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    curve = TideCurve([TideSample(start, 1.0), TideSample(start + timedelta(hours=6), 2.5)])
    req = UnderKeelRequirement(12.0, 11.5, 1.0)
    at = start + timedelta(hours=3)
    return {'project': 'TerminalWeave', 'tide_height_m': round(curve.height_at(at), 3), 'clearance_m': round(curve.clearance_m(at, req), 3), 'safe': curve.is_safe(at, req)}

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(prog='terminalweave')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('demo')
    sub.add_parser('arrivals')
    http = sub.add_parser('serve')
    http.add_argument('--host', default='127.0.0.1')
    http.add_argument('--port', type=int, default=8080)
    args = parser.parse_args(argv)
    if args.command == 'demo':
        print(json.dumps(demo(), sort_keys=True))
        return 0
    if args.command == 'arrivals':
        print(json.dumps(arrival_demo(), sort_keys=True, default=str))
        return 0
    if args.command == 'serve':
        serve(args.host, args.port)
        return 0
    return 2
if __name__ == '__main__':
    raise SystemExit(main())
