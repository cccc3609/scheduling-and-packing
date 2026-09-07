"""Single authoritative schema for SchedulingEnv observations."""

from dataclasses import dataclass
from enum import IntEnum


SCHED_DYNAMIC_GLOBAL_DIM = 3
SCHED_CONTEXT_DIM = 8
SCHED_TASK_DIM = 10


class SchedulingTaskFeature(IntEnum):
    PROCESSING_TIME = 0
    RELATIVE_DUE = 1
    PLATE_AREA_RATIO = 2
    LINKED_ORDER_RATIO = 3
    SCHEDULED = 4
    MEAN_REMAINING_RATIO = 5
    MAX_REMAINING_RATIO = 6
    RELEASE_FRACTION = 7
    WORST_SLACK = 8
    VALID = 9


SCHED_TASK_VALID_INDEX = int(SchedulingTaskFeature.VALID)


@dataclass(frozen=True)
class SchedulingObservationLayout:
    """Offsets for ``machines + globals + context + task tokens``."""

    num_machines: int
    max_tasks: int

    def __post_init__(self):
        if self.num_machines < 1:
            raise ValueError("num_machines must be positive")
        if self.max_tasks < 1:
            raise ValueError("max_tasks must be positive")

    @property
    def dynamic_global_dim(self) -> int:
        return SCHED_DYNAMIC_GLOBAL_DIM

    @property
    def context_dim(self) -> int:
        return SCHED_CONTEXT_DIM

    @property
    def task_dim(self) -> int:
        return SCHED_TASK_DIM

    @property
    def task_valid_index(self) -> int:
        return SCHED_TASK_VALID_INDEX

    @property
    def machine_slice(self) -> slice:
        return slice(0, self.num_machines)

    @property
    def dynamic_global_slice(self) -> slice:
        start = self.machine_slice.stop
        return slice(start, start + self.dynamic_global_dim)

    @property
    def context_slice(self) -> slice:
        start = self.dynamic_global_slice.stop
        return slice(start, start + self.context_dim)

    @property
    def global_prefix_dim(self) -> int:
        return self.context_slice.stop

    @property
    def task_slice(self) -> slice:
        start = self.global_prefix_dim
        return slice(start, start + self.max_tasks * self.task_dim)

    @property
    def obs_dim(self) -> int:
        return self.task_slice.stop

    def task_slot_slice(self, task_index: int) -> slice:
        if not 0 <= task_index < self.max_tasks:
            raise IndexError("task_index outside observation capacity")
        start = self.task_slice.start + task_index * self.task_dim
        return slice(start, start + self.task_dim)

