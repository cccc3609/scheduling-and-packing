"""Single authoritative schema for nesting observations."""

from dataclasses import dataclass
from enum import IntEnum

DEFAULT_NESTING_MAX_PARTS = 120
NESTING_GLOBAL_DIM = 21
NESTING_SKYLINE_DIM = 20
NESTING_CONTEXT_DIM = 16
NESTING_PART_DIM = 6


class NestingPartFeature(IntEnum):
    WIDTH = 0
    HEIGHT = 1
    AREA = 2
    DUE_SLACK = 3
    PACKED = 4
    VALID = 5


NESTING_PART_PACKED_INDEX = int(NestingPartFeature.PACKED)
NESTING_PART_VALID_INDEX = int(NestingPartFeature.VALID)

LEGACY_NESTING_SCHEMA_ERROR = (
    "Legacy nesting checkpoint uses the 5-D token / 657-D observation schema. "
    "Patch 5 requires explicit 6-D validity tokens and retraining."
)


@dataclass(frozen=True)
class NestingObservationLayout:
    """Offsets for ``part tokens + globals + skyline + context``."""

    max_parts: int = DEFAULT_NESTING_MAX_PARTS
    part_dim: int = NESTING_PART_DIM
    global_dim: int = NESTING_GLOBAL_DIM
    skyline_dim: int = NESTING_SKYLINE_DIM
    context_dim: int = NESTING_CONTEXT_DIM

    def __post_init__(self):
        for name in ("max_parts", "part_dim", "global_dim", "skyline_dim", "context_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.part_dim <= self.valid_index:
            raise ValueError("part_dim does not contain the canonical VALID feature")

    @property
    def packed_index(self) -> int:
        return NESTING_PART_PACKED_INDEX

    @property
    def valid_index(self) -> int:
        return NESTING_PART_VALID_INDEX

    @property
    def part_block_dim(self) -> int:
        return self.max_parts * self.part_dim

    @property
    def state_dim(self) -> int:
        return self.global_dim + self.skyline_dim + self.context_dim

    @property
    def part_slice(self) -> slice:
        return slice(0, self.part_block_dim)

    @property
    def global_slice(self) -> slice:
        start = self.part_slice.stop
        return slice(start, start + self.global_dim)

    @property
    def skyline_slice(self) -> slice:
        start = self.global_slice.stop
        return slice(start, start + self.skyline_dim)

    @property
    def context_slice(self) -> slice:
        start = self.skyline_slice.stop
        return slice(start, start + self.context_dim)

    @property
    def state_slice(self) -> slice:
        return slice(self.global_slice.start, self.context_slice.stop)

    @property
    def obs_dim(self) -> int:
        return self.context_slice.stop

    def part_slot_slice(self, part_index: int) -> slice:
        if not 0 <= part_index < self.max_parts:
            raise IndexError("part_index outside observation capacity")
        start = self.part_slice.start + part_index * self.part_dim
        return slice(start, start + self.part_dim)

    def validate_observation_width(self, width: int) -> None:
        if int(width) != self.obs_dim:
            raise ValueError(
                f"Expected nesting observation width {self.obs_dim}, got {width}. "
                f"{LEGACY_NESTING_SCHEMA_ERROR}"
            )


def validate_nesting_checkpoint_observation_space(model, layout=None) -> None:
    """Reject SB3 nesting policies saved against a non-canonical observation."""
    layout = layout or NestingObservationLayout()
    observation_space = getattr(model, "observation_space", None)
    shape = getattr(observation_space, "shape", None)
    if not shape or len(shape) != 1:
        raise ValueError("Nesting checkpoint has no one-dimensional observation space")
    layout.validate_observation_width(shape[0])
