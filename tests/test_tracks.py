from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from skyweaver.observatories import Observatory
from skyweaver.orbits import OrbitSpec
from skyweaver.timegrid import TimeGrid
from skyweaver.tracks import GroundTrack, SkyTrack, ground_track, sky_track


def make_test_orbit() -> OrbitSpec:
    """Return a simple circular test orbit."""
    return OrbitSpec.circular(
        name="test_sat",
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        altitude_km=550.0,
        inclination_deg=70.0,
        raan_deg=30.0,
        phase_deg=120.0,
    )


def make_test_timegrid() -> TimeGrid:
    """Return a short test time grid."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stop = start + timedelta(minutes=10)
    return TimeGrid(start=start, stop=stop, cadence_s=60.0)


def test_ground_track_returns_groundtrack_instance() -> None:
    track = ground_track(make_test_orbit(), make_test_timegrid())
    assert isinstance(track, GroundTrack)


def test_ground_track_array_lengths_match_timegrid() -> None:
    timegrid = make_test_timegrid()
    track = ground_track(make_test_orbit(), timegrid)

    assert len(track.latitude_deg) == timegrid.n_times
    assert len(track.longitude_deg) == timegrid.n_times
    assert len(track.elevation_m) == timegrid.n_times
    assert track.n_times == timegrid.n_times


def test_ground_track_lat_lon_ranges_are_reasonable() -> None:
    track = ground_track(make_test_orbit(), make_test_timegrid())

    assert np.all(track.latitude_deg >= -90.0)
    assert np.all(track.latitude_deg <= 90.0)
    assert np.all(track.longitude_deg >= -180.0)
    assert np.all(track.longitude_deg <= 180.0)


def test_ground_track_summary_contains_key_fields() -> None:
    track = ground_track(make_test_orbit(), make_test_timegrid())
    summary = track.summary()

    assert "GroundTrack(" in summary
    assert "test_sat" in summary
    assert f"n_times={track.n_times}" in summary


def test_sky_track_returns_skytrack_instance() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )
    assert isinstance(track, SkyTrack)


def test_sky_track_array_lengths_match_timegrid() -> None:
    timegrid = make_test_timegrid()
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        timegrid,
    )

    assert len(track.altitude_deg) == timegrid.n_times
    assert len(track.azimuth_deg) == timegrid.n_times
    assert len(track.range_km) == timegrid.n_times
    assert track.n_times == timegrid.n_times


def test_sky_track_ranges_are_reasonable() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )

    assert np.all(track.altitude_deg >= -90.0)
    assert np.all(track.altitude_deg <= 90.0)
    assert np.all(track.azimuth_deg >= 0.0)
    assert np.all(track.azimuth_deg <= 360.0)
    assert np.all(track.range_km > 0.0)


def test_sky_track_visible_is_boolean_and_correct_shape() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )

    assert track.visible.dtype == np.bool_
    assert len(track.visible) == track.n_times
    assert np.array_equal(track.visible, track.altitude_deg > 0.0)


def test_sky_track_summary_contains_key_fields() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )
    summary = track.summary()

    assert "SkyTrack(" in summary
    assert "test_sat" in summary
    assert "MWA" in summary
    assert f"n_times={track.n_times}" in summary
