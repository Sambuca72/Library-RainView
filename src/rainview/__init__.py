"""RainView — воспроизводимые raincloud plots из pandas DataFrame."""

from .exceptions import (
    ColumnNotFoundError,
    ConstantGroupWarning,
    EmptyGroupError,
    GroupCountWarning,
    InsufficientDataError,
    InsufficientGroupsError,
    InvalidDataError,
    InvalidParameterError,
    LargeDatasetWarning,
    MissingDataWarning,
    NonNumericColumnError,
    RainViewError,
    RainViewWarning,
    SmallSampleWarning,
)
from .plotting import raincloudplot
from .reporting import format_summary, save_plot

__version__ = "0.1.0"
__all__ = [
    "raincloudplot",
    "format_summary",
    "save_plot",
    "RainViewError",
    "RainViewWarning",
    "ColumnNotFoundError",
    "NonNumericColumnError",
    "EmptyGroupError",
    "InsufficientDataError",
    "InsufficientGroupsError",
    "InvalidDataError",
    "InvalidParameterError",
    "SmallSampleWarning",
    "MissingDataWarning",
    "ConstantGroupWarning",
    "GroupCountWarning",
    "LargeDatasetWarning",
]
