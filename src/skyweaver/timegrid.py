"""Time grid utilities for satellite simulations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from skyfield.api import Time, load


def _ensure_utc(dt: datetime) -> datetime:
    """Return a timezone-aware UTC datetime.

    Parameters
    ----------
    dt
        Input datetime. Naive datetimes are assumed to already be in UTC.

    Returns
    -------
    datetime
        Timezone-aware UTC datetime.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class TimeGrid:
    """Regular time sampling grid.

    Parameters
    ----------
    start
        Start datetime of the grid. Naive datetimes are interpreted as UTC.
    stop
        Stop datetime of the grid. Naive datetimes are interpreted as UTC.
    cadence_s
        Sampling cadence in seconds. Must be positive.

    Notes
    -----
    The generated grid includes the start time and includes the stop time only
    if it lies exactly on the cadence.
    """

    start: datetime
    stop: datetime
    cadence_s: float

    def __post_init__(self) -> None:
        """Validate inputs and normalize datetimes to UTC."""
        start_utc = _ensure_utc(self.start)
        stop_utc = _ensure_utc(self.stop)

        object.__setattr__(self, "start", start_utc)
        object.__setattr__(self, "stop", stop_utc)

        if self.cadence_s <= 0.0:
            raise ValueError("cadence_s must be positive")

        if self.stop < self.start:
            raise ValueError("stop must be greater than or equal to start")

    @property
    def duration_s(self) -> float:
        """Return the total duration of the grid in seconds."""
        return (self.stop - self.start).total_seconds()

    @property
    def n_times(self) -> int:
        """Return the number of samples in the grid."""
        return int(np.floor(self.duration_s / self.cadence_s)) + 1

    @property
    def offsets_s(self) -> np.ndarray:
        """Return sample offsets from start in seconds."""
        return np.arange(self.n_times, dtype=float) * self.cadence_s

    def datetimes(self) -> list[datetime]:
        """Return the grid as a list of UTC datetimes."""
        return [self.start + timedelta(seconds=float(offset_s)) for offset_s in self.offsets_s]

    def skyfield(self) -> Time:
        """Return the grid as a Skyfield ``Time`` array."""
        ts = load.timescale()
        return ts.from_datetimes(self.datetimes())

    def summary(self) -> str:
        """Return a compact human-readable summary."""
        return (
            f"TimeGrid(start={self.start.isoformat()}, "
            f"stop={self.stop.isoformat()}, "
            f"cadence_s={self.cadence_s:.3f}, "
            f"n_times={self.n_times})"
        )
