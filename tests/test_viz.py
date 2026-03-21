from __future__ import annotations

from datetime import datetime, timedelta, timezone

import matplotlib

matplotlib.use("Agg")

import healpy as hp
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.projections.polar import PolarAxes

from skyweaver.observatories import Observatory
from skyweaver.orbits import OrbitSpec
from skyweaver.timegrid import TimeGrid
from skyweaver.tracks import PassInterval, SkyPass, SkyTrack, ground_track, sky_track
from skyweaver.viz import plot_ground_track, plot_sky_track, plot_sky_track_healpix
from skyweaver.viz.sky import _plot_single_pass


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


def make_test_interval() -> PassInterval:
    return PassInterval(
        start_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        stop_utc=datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
    )


def make_test_sky_pass() -> SkyPass:
    return SkyPass(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        interval=make_test_interval(),
        cadence_s=1.0,
        start_index=0,
        stop_index=3,
        altitude_deg=np.array([10.0, 25.0, 40.0]),
        azimuth_deg=np.array([180.0, 200.0, 220.0]),
        range_km=np.array([1200.0, 1100.0, 1000.0]),
    )


def test_plot_ground_track_returns_axes() -> None:
    track = ground_track(make_test_orbit(), make_test_timegrid())
    ax = plot_ground_track(track)

    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Longitude [deg]"
    assert ax.get_ylabel() == "Latitude [deg]"


def test_plot_ground_track_with_observatories_and_labels() -> None:
    track = ground_track(make_test_orbit(), make_test_timegrid())

    ax = plot_ground_track(
        track,
        observatories=[
            Observatory.get("MWA"),
            Observatory.get("LOFAR"),
        ],
        show_labels=True,
    )

    assert isinstance(ax, Axes)
    assert "Ground track" in ax.get_title()


def test_plot_ground_track_with_observatories_without_labels() -> None:
    track = ground_track(make_test_orbit(), make_test_timegrid())

    ax = plot_ground_track(
        track,
        observatories=[
            Observatory.get("MWA"),
            Observatory.get("LOFAR"),
        ],
        show_labels=False,
    )

    assert isinstance(ax, Axes)


def test_plot_sky_track_returns_axes() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )
    ax = plot_sky_track(track)

    assert isinstance(ax, PolarAxes)
    assert "Sky track" in ax.get_title()


def test_plot_single_pass_returns_none_and_plots() -> None:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    _plot_single_pass(make_test_sky_pass(), ax)

    assert isinstance(ax, PolarAxes)


def test_plot_single_pass_handles_empty_pass() -> None:
    import matplotlib.pyplot as plt

    empty_pass = SkyPass(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        interval=PassInterval(
            start_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            stop_utc=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ),
        cadence_s=1.0,
        start_index=0,
        stop_index=0,
        altitude_deg=np.array([]),
        azimuth_deg=np.array([]),
        range_km=np.array([]),
    )

    _, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    _plot_single_pass(empty_pass, ax)

    assert isinstance(ax, PolarAxes)


def test_plot_sky_track_accepts_color() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )
    ax = plot_sky_track(track, color="seagreen", alpha=0.5)

    assert isinstance(ax, PolarAxes)


def test_plot_sky_track_accepts_cmap_string() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )
    ax = plot_sky_track(track, cmap="viridis", alpha=0.5)

    assert isinstance(ax, PolarAxes)


def test_plot_sky_track_accepts_cmap_object() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )
    ax = plot_sky_track(track, cmap=colormaps["viridis"], alpha=0.5)

    assert isinstance(ax, PolarAxes)


def test_plot_sky_track_handles_no_visible_passes() -> None:
    track = SkyTrack(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        altitude_deg=np.array([-10.0, -5.0, 0.0]),
        azimuth_deg=np.array([0.0, 45.0, 90.0]),
        range_km=np.array([1000.0, 1001.0, 1002.0]),
        cadence_s=60.0,
        timegrid=make_test_timegrid(),
    )

    ax = plot_sky_track(track, cmap="viridis")

    assert isinstance(ax, PolarAxes)


def test_plot_sky_track_single_pass_cmap_branch() -> None:
    track = SkyTrack(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        altitude_deg=np.array([-5.0, 20.0, 30.0, -1.0]),
        azimuth_deg=np.array([0.0, 10.0, 20.0, 30.0]),
        range_km=np.array([1000.0, 1001.0, 1002.0, 1003.0]),
        cadence_s=60.0,
        timegrid=make_test_timegrid(),
    )

    ax = plot_sky_track(track, cmap="viridis")

    assert isinstance(ax, PolarAxes)


def test_plot_sky_track_multiple_passes_cmap_branch() -> None:
    track = SkyTrack(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        altitude_deg=np.array([-5.0, 20.0, 30.0, -1.0, 25.0, 35.0, -2.0]),
        azimuth_deg=np.array([0.0, 10.0, 20.0, 30.0, 180.0, 190.0, 200.0]),
        range_km=np.array([1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0]),
        cadence_s=60.0,
        timegrid=make_test_timegrid(),
    )

    ax = plot_sky_track(track, cmap="viridis")

    assert isinstance(ax, PolarAxes)


def test_plot_sky_track_healpix_returns_map() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )

    hmap = plot_sky_track_healpix(
        track,
        nside=8,
        graticule=False,
    )

    assert hmap.shape == (hp.nside2npix(8),)
    assert np.all(hmap >= 0.0)


def test_plot_sky_track_healpix_with_title_and_graticule() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )

    hmap = plot_sky_track_healpix(
        track,
        nside=8,
        title="Custom title",
        graticule=True,
    )

    assert hmap.shape == (hp.nside2npix(8),)
    assert np.all(hmap >= 0.0)
