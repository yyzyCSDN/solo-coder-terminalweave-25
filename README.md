# TerminalWeave

TerminalWeave is a dependency-light Python engine for modelling day-to-day container-terminal operations. It covers marine access, berth use, quay cranes, yard placement, truck-gate appointments, customs holds, reefer monitoring, hazardous-goods segregation, equipment maintenance, rail blocks, demurrage billing and transshipment connections.

The package is intentionally usable without a database or web framework. Domain services expose ordinary Python APIs, while the persistence layer supplies an append-only event store and snapshots for embedding in larger systems.

## Quick start

```bash
python -m pip install -e ".[test]"
python -m terminalweave.cli demo
python -m pytest -q
```

Run the small HTTP adapter with `terminalweave serve`. It exposes `GET /health`,
`POST /tide/clearance`, and `POST /billing/demurrage`. The other packages are
library components intended to be composed by an embedding application.

## Arrival planning

`marine.arrival` chains the otherwise independent tide, weather, channel,
pilotage, tug and berth judgments into queryable arrival plans
(`terminalweave arrivals` runs a worked scenario):

- every gate must pass and all required information must be present — any
  missing forecast or failed safety gate keeps the call un-released and the
  plan records a structured `GateFinding` reason per gate;
- `update_forecast` (new revision) and `resource_changed` (roster, fleet or
  berth-closure change) only recompute *unconfirmed* plans; confirmed and
  executed records, their windows and their history stay untouched;
- each rejection and reschedule in the returned `ReplanReport` carries the
  gate findings (e.g. `tide/insufficient_ukc`, `pilot/no_qualified_pilot`,
  `tug/requirement_unmet`, `berth/no_berth`) that explain the decision.

## Design principles

- domain calculations are deterministic for identical inputs;
- time values are timezone-aware UTC datetimes;
- operational state transitions are explicit and validated;
- safety constraints fail closed when required information is absent;
- money uses `Decimal`;
- core domain modules depend only on the Python standard library.

## Main packages

`marine`, `cranes`, `yard`, `gate`, `customs`, `reefer`, `hazmat`, `equipment`, `rail`, `billing`, `transshipment`, `planning`, `persistence`, `reporting`, and `api`.
