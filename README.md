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

## Design principles

- domain calculations are deterministic for identical inputs;
- time values are timezone-aware UTC datetimes;
- operational state transitions are explicit and validated;
- safety constraints fail closed when required information is absent;
- money uses `Decimal`;
- core domain modules depend only on the Python standard library.

## Main packages

`marine`, `cranes`, `yard`, `gate`, `customs`, `reefer`, `hazmat`, `equipment`, `rail`, `billing`, `transshipment`, `planning`, `persistence`, `reporting`, and `api`.

### Chained port-entry plans

`marine.entry_plan.PortEntryPlanner` composes the standalone tide, weather,
channel, pilotage, tug and berth judgements into one queryable entry plan:

- `submit(EntryRequest)` searches the requested entry window on a fixed grid and
  returns the earliest slot that clears every check, or a blocked decision with
  the full list of `DenyReason` codes;
- the gate **fails closed** — a missing tide curve for any channel leg, a
  weather forecast that does not cover the transit (or a sample gap beyond the
  configured cadence), or an absent roster/fleet/berth/route dataset blocks the
  vessel just like a violated limit;
- `replan_unconfirmed(at)` reruns the search after a forecast or resource
  change, touching only `proposed` plans. `confirmed` and `executed` records are
  immutable and still reserve their pilot, tugs and berth space;
- every rejection, postponement, confirmation and execution is appended to the
  plan history with its reasons, the prior slot and the revision;
- `confirm` revalidates the slot before freezing it, `execute` follows
  confirmation, and `explain` dry-runs a single candidate without mutating state.
