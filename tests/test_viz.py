from __future__ import annotations

from datetime import datetime, timedelta, timezone

import matplotlib

matplotlib.use("Agg")

from matplotlib.axes import Axes
from matplotlib.projections.polar import PolarAxes

from skyweaver.observatories import Observatory
from skyweaver.orbits import OrbitSpec
from skyweaver.timegrid import TimeGrid
from skyweaver.tracks import ground_track, sky_track
from skyweaver.viz import plot_ground_track, plot_sky_track


def make_test_orbit() -> OrbitSpec:
    return OrbitSpec.circular(
        name="test_sat",
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        altitude_km=550.0,
        inclination_deg=70.0,
        raan_deg=30.0,
        phase_deg=120.0,
    )


def make_test_timegrid() -> TimeGrid:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stop = start + timedelta(minutes=10)
    return TimeGrid(start=start, stop=stop, cadence_s=60.0)


def test_plot_ground_track_returns_axes() -> None:
    track = ground_track(make_test_orbit(), make_test_timegrid())
    ax = plot_ground_track(track)

    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Longitude [deg]"
    assert ax.get_ylabel() == "Latitude [deg]"


def test_plot_sky_track_returns_axes() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )
    ax = plot_sky_track(track)

    assert isinstance(ax, PolarAxes)
    assert "Sky track" in ax.get_title()
