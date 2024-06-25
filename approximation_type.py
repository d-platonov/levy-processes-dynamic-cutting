from enum import Enum


class ApproximationType(Enum):
    """Enum for specifying the type of approximation applied to the Lévy process."""
    CUT_1 = 1
    CUT_2 = 2
