from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError, ConflictError
from terminalweave.core.intervals import TimeInterval

@dataclass(frozen=True)
class Berth:
    berth_id: str
    quay_length_m: float
    max_draft_m: float
    bollard_spacing_m: float = 25.0

    def __post_init__(self) -> None:
        if self.quay_length_m <= 0 or self.max_draft_m <= 0 or self.bollard_spacing_m <= 0:
            raise ValidationError('berth dimensions must be positive')

@dataclass(frozen=True)
class VesselCall:
    call_id: str
    loa_m: float
    beam_m: float
    arrival: datetime
    departure: datetime
    draft_m: float
    preferred_position_m: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'arrival', ensure_utc(self.arrival))
        object.__setattr__(self, 'departure', ensure_utc(self.departure))
        if self.departure <= self.arrival:
            raise ValidationError('departure must follow arrival')
        if min(self.loa_m, self.beam_m, self.draft_m) <= 0:
            raise ValidationError('vessel dimensions must be positive')

    @property
    def interval(self) -> TimeInterval:
        return TimeInterval(self.arrival, self.departure)

@dataclass(frozen=True)
class BerthAssignment:
    berth_id: str
    call_id: str
    interval: TimeInterval
    start_m: float
    end_m: float
    safety_margin_m: float

    def overlaps_space(self, other: 'BerthAssignment') -> bool:
        return self.start_m < other.end_m and other.start_m < self.end_m

    def conflicts(self, other: 'BerthAssignment') -> bool:
        return self.berth_id == other.berth_id and self.interval.overlaps(other.interval) and self.overlaps_space(other)

class BerthPlanner:

    def __init__(self, berth: Berth, safety_margin_m: float=15.0):
        if safety_margin_m < 0:
            raise ValidationError('safety margin cannot be negative')
        self.berth = berth
        self.safety_margin_m = safety_margin_m
        self._assignments: list[BerthAssignment] = []

    @property
    def assignments(self) -> tuple[BerthAssignment, ...]:
        return tuple(sorted(self._assignments, key=lambda a: (a.interval.start, a.start_m)))

    def _required_length(self, vessel: VesselCall) -> float:
        return vessel.loa_m + 2 * self.safety_margin_m

    def _candidate_positions(self, vessel: VesselCall) -> list[float]:
        required = self._required_length(vessel)
        max_start = self.berth.quay_length_m - required
        if max_start < 0:
            return []
        candidates = {0.0, max_start}
        if vessel.preferred_position_m is not None:
            preferred = max(0.0, min(vessel.preferred_position_m - self.safety_margin_m, max_start))
            candidates.add(preferred)
        for assignment in self._assignments:
            candidates.add(max(0.0, min(assignment.end_m, max_start)))
            candidates.add(max(0.0, min(assignment.start_m - required, max_start)))
        return sorted(candidates, key=lambda x: (abs(x + required / 2 - (vessel.preferred_position_m or required / 2)), x))

    def can_place(self, vessel: VesselCall, start_m: float) -> bool:
        if vessel.draft_m > self.berth.max_draft_m:
            return False
        required = self._required_length(vessel)
        if start_m < 0 or start_m + required > self.berth.quay_length_m:
            return False
        candidate = BerthAssignment(self.berth.berth_id, vessel.call_id, vessel.interval, start_m, start_m + required, self.safety_margin_m)
        return all((not candidate.conflicts(existing) for existing in self._assignments))

    def first_fit_position(self, vessel: VesselCall) -> float | None:
        if vessel.draft_m > self.berth.max_draft_m:
            return None
        for start in self._candidate_positions(vessel):
            if self.can_place(vessel, start):
                return start
        return None

    def assign(self, vessel: VesselCall) -> BerthAssignment:
        start = self.first_fit_position(vessel)
        if start is None:
            if vessel.draft_m > self.berth.max_draft_m:
                raise ConflictError('vessel draft exceeds berth limit')
            raise ConflictError('no conflict-free berth position is available')
        required = self._required_length(vessel)
        assignment = BerthAssignment(self.berth.berth_id, vessel.call_id, vessel.interval, start, start + required, self.safety_margin_m)
        self._assignments.append(assignment)
        return assignment

    def unassign(self, call_id: str) -> bool:
        before = len(self._assignments)
        self._assignments = [a for a in self._assignments if a.call_id != call_id]
        return len(self._assignments) != before

    def utilization(self, interval: TimeInterval) -> float:
        if interval.duration.total_seconds() <= 0:
            return 0.0
        occupied_area = 0.0
        for assignment in self._assignments:
            overlap = assignment.interval.intersection(interval)
            if overlap is None:
                continue
            occupied_area += (assignment.end_m - assignment.start_m) * overlap.duration.total_seconds()
        capacity_area = self.berth.quay_length_m * interval.duration.total_seconds()
        return min(1.0, occupied_area / capacity_area)
