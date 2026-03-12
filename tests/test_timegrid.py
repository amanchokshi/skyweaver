from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from skyweaver.timegrid import TimeGrid, _ensure_utc


def test_ensure_utc_converts_naive_datetime() -> None:
    dt = datetime(2026, 1, 1, 12, 0, 0)
    out = _ensure_utc(dt)

    assert out.tzinfo == timezone.utc
    assert out == datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_ensure_utc_preserves_aware_utc_datetime() -> None:
    dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    out = _ensure_utc(dt)

    assert out.tzinfo == timezone.utc
    assert out == dt


def test_timegrid_rejects_non_positive_cadence() -> None:
    with pytest.raises(ValueError, match="cadence_s must be positive"):
        TimeGrid(
            start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            stop=datetime(2026, 1, 2, tzinfo=timezone.utc),
            cadence_s=0.0,
        )


def test_timegrid_rejects_stop_before_start() -> None:
    with pytest.raises(
        ValueError,
        match="stop must be greater than or equal to start",
    ):
        TimeGrid(
            start=datetime(2026, 1, 2, tzinfo=timezone.utc),
            stop=datetime(2026, 1, 1, tzinfo=timezone.utc),
            cadence_s=10.0,
        )


def test_timegrid_allows_single_sample() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    grid = TimeGrid(
        start=start,
        stop=start,
        cadence_s=10.0,
    )

    assert grid.duration_s == 0.0
    assert grid.n_times == 1
    assert grid.datetimes() == [start]


def test_timegrid_duration_and_offsets() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stop = start + timedelta(seconds=30)

    grid = TimeGrid(
        start=start,
        stop=stop,
        cadence_s=10.0,
    )

    assert grid.duration_s == 30.0
    assert grid.n_times == 4
    assert np.allclose(grid.offsets_s, [0.0, 10.0, 20.0, 30.0])


def test_timegrid_datetimes_exact_stop_on_grid() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stop = start + timedelta(seconds=20)

    grid = TimeGrid(
        start=start,
        stop=stop,
        cadence_s=10.0,
    )

    dts = grid.datetimes()

    assert len(dts) == 3
    assert dts[0] == start
    assert dts[1] == start + timedelta(seconds=10)
    assert dts[2] == stop


def test_timegrid_datetimes_stop_not_on_grid() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stop = start + timedelta(seconds=25)

    grid = TimeGrid(
        start=start,
        stop=stop,
        cadence_s=10.0,
    )

    dts = grid.datetimes()

    assert len(dts) == 3
    assert dts[0] == start
    assert dts[1] == start + timedelta(seconds=10)
    assert dts[2] == start + timedelta(seconds=20)


def test_timegrid_skyfield_length_matches_n_times() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stop = start + timedelta(seconds=20)

    grid = TimeGrid(
        start=start,
        stop=stop,
        cadence_s=10.0,
    )

    t = grid.skyfield()

    assert len(t) == grid.n_times


def test_timegrid_normalizes_naive_datetimes_to_utc() -> None:
    grid = TimeGrid(
        start=datetime(2026, 1, 1, 0, 0, 0),
        stop=datetime(2026, 1, 1, 0, 0, 20),
        cadence_s=10.0,
    )

    assert grid.start.tzinfo == timezone.utc
    assert grid.stop.tzinfo == timezone.utc


def test_timegrid_summary_contains_key_fields() -> None:
    grid = TimeGrid(
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        stop=datetime(2026, 1, 1, 0, 0, 20, tzinfo=timezone.utc),
        cadence_s=10.0,
    )

    summary = grid.summary()

    assert "TimeGrid(" in summary
    assert "cadence_s=10.000" in summary
    assert "n_times=3" in summary
