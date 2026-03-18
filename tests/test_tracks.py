from __future__ import annotations

from datetime import datetime, timedelta, timezone

import healpy as hp
import numpy as np

from skyweaver.observatories import Observatory
from skyweaver.orbits import OrbitSpec
from skyweaver.timegrid import TimeGrid
from skyweaver.tracks import GroundTrack, SkyPass, SkyTrack, ground_track, sky_track


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


def make_test_sky_pass() -> SkyPass:
    """Return a simple SkyPass instance for direct property testing."""
    return SkyPass(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        timegrid=make_test_timegrid(),
        start_index=2,
        stop_index=5,
        altitude_deg=np.array([10.0, 25.0, 40.0]),
        azimuth_deg=np.array([180.0, 200.0, 220.0]),
        range_km=np.array([1200.0, 1100.0, 1000.0]),
    )


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


def test_sky_pass_n_times_returns_array_length() -> None:
    sat_pass = make_test_sky_pass()

    assert sat_pass.n_times == 3


def test_sky_pass_max_altitude_deg_returns_maximum() -> None:
    sat_pass = make_test_sky_pass()

    assert sat_pass.max_altitude_deg == 40.0


def test_sky_pass_stores_metadata_correctly() -> None:
    sat_pass = make_test_sky_pass()

    assert sat_pass.orbit.name == "test_sat"
    assert sat_pass.observatory.name == "MWA"
    assert sat_pass.start_index == 2
    assert sat_pass.stop_index == 5


def test_sky_pass_arrays_are_preserved() -> None:
    sat_pass = make_test_sky_pass()

    assert np.array_equal(sat_pass.altitude_deg, np.array([10.0, 25.0, 40.0]))
    assert np.array_equal(sat_pass.azimuth_deg, np.array([180.0, 200.0, 220.0]))
    assert np.array_equal(sat_pass.range_km, np.array([1200.0, 1100.0, 1000.0]))


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


def test_sky_track_passes_returns_list() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )

    passes = track.passes()

    assert isinstance(passes, list)
    assert all(isinstance(sat_pass, SkyPass) for sat_pass in passes)


def test_sky_passes_are_above_horizon() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )

    for sat_pass in track.passes():
        assert np.all(sat_pass.altitude_deg > 0.0)


def test_sky_passes_partition_visible_samples() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )

    total_pass_samples = sum(sat_pass.n_times for sat_pass in track.passes())
    assert total_pass_samples == int(np.sum(track.visible))


def test_sky_pass_indices_match_parent_track_slices() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )

    for sat_pass in track.passes():
        start = sat_pass.start_index
        stop = sat_pass.stop_index

        assert np.array_equal(sat_pass.altitude_deg, track.altitude_deg[start:stop])
        assert np.array_equal(sat_pass.azimuth_deg, track.azimuth_deg[start:stop])
        assert np.array_equal(sat_pass.range_km, track.range_km[start:stop])


def test_sky_pass_n_times_matches_array_length() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )

    for sat_pass in track.passes():
        assert sat_pass.n_times == len(sat_pass.altitude_deg)
        assert sat_pass.n_times == len(sat_pass.azimuth_deg)
        assert sat_pass.n_times == len(sat_pass.range_km)


def test_sky_pass_max_altitude_deg_is_correct() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )

    for sat_pass in track.passes():
        assert np.isclose(sat_pass.max_altitude_deg, np.max(sat_pass.altitude_deg))


def test_sky_track_passes_returns_empty_list_when_never_visible() -> None:
    track = SkyTrack(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        timegrid=make_test_timegrid(),
        altitude_deg=np.array([-10.0, -5.0, 0.0, -1.0]),
        azimuth_deg=np.array([0.0, 10.0, 20.0, 30.0]),
        range_km=np.array([1000.0, 1001.0, 1002.0, 1003.0]),
    )

    assert track.passes() == []


def test_sky_track_passes_handles_visible_from_first_sample() -> None:
    track = SkyTrack(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        timegrid=make_test_timegrid(),
        altitude_deg=np.array([5.0, 10.0, -1.0, -2.0]),
        azimuth_deg=np.array([0.0, 10.0, 20.0, 30.0]),
        range_km=np.array([1000.0, 1001.0, 1002.0, 1003.0]),
    )

    passes = track.passes()

    assert len(passes) == 1
    assert passes[0].start_index == 0
    assert passes[0].stop_index == 2


def test_sky_track_passes_handles_visible_until_last_sample() -> None:
    track = SkyTrack(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        timegrid=make_test_timegrid(),
        altitude_deg=np.array([-5.0, -1.0, 2.0, 8.0]),
        azimuth_deg=np.array([0.0, 10.0, 20.0, 30.0]),
        range_km=np.array([1000.0, 1001.0, 1002.0, 1003.0]),
    )

    passes = track.passes()

    assert len(passes) == 1
    assert passes[0].start_index == 2
    assert passes[0].stop_index == 4


def test_sky_track_passes_handles_multiple_passes() -> None:
    track = SkyTrack(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        timegrid=make_test_timegrid(),
        altitude_deg=np.array([-5.0, 2.0, 3.0, -1.0, 4.0, 5.0, -2.0]),
        azimuth_deg=np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        range_km=np.array([1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0]),
    )

    passes = track.passes()

    assert len(passes) == 2

    assert passes[0].start_index == 1
    assert passes[0].stop_index == 3
    assert np.array_equal(passes[0].altitude_deg, np.array([2.0, 3.0]))

    assert passes[1].start_index == 4
    assert passes[1].stop_index == 6
    assert np.array_equal(passes[1].altitude_deg, np.array([4.0, 5.0]))


def test_sky_track_to_healpix_returns_correct_size() -> None:
    track = sky_track(
        make_test_orbit(),
        Observatory.get("MWA"),
        make_test_timegrid(),
    )

    nside = 8
    hmap = track.to_healpix(nside=nside)

    assert hmap.shape == (hp.nside2npix(nside),)
    assert np.all(hmap >= 0.0)


def test_sky_track_to_healpix_returns_zeros_when_no_visible_passes() -> None:
    track = SkyTrack(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        timegrid=make_test_timegrid(),
        altitude_deg=np.array([-10.0, -5.0, 0.0]),
        azimuth_deg=np.array([0.0, 45.0, 90.0]),
        range_km=np.array([1000.0, 1001.0, 1002.0]),
    )

    hmap = track.to_healpix(nside=8)

    assert np.all(hmap == 0.0)


def test_sky_track_to_healpix_unique_per_pass_changes_counts() -> None:
    track = SkyTrack(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        timegrid=make_test_timegrid(),
        altitude_deg=np.array([30.0, 30.0, 30.0]),
        azimuth_deg=np.array([45.0, 45.0, 45.0]),
        range_km=np.array([1000.0, 1000.0, 1000.0]),
    )

    hmap_unique = track.to_healpix(nside=8, unique_per_pass=True)
    hmap_all = track.to_healpix(nside=8, unique_per_pass=False)

    assert np.sum(hmap_unique) == 1.0
    assert np.sum(hmap_all) == 3.0


def test_sky_track_to_healpix_counts_distinct_pixels_within_pass() -> None:
    track = SkyTrack(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        timegrid=make_test_timegrid(),
        altitude_deg=np.array([30.0, 31.0, 32.0]),
        azimuth_deg=np.array([45.0, 60.0, 75.0]),
        range_km=np.array([1000.0, 1000.0, 1000.0]),
    )

    hmap = track.to_healpix(nside=32, unique_per_pass=True)

    assert np.sum(hmap) >= 1.0
    assert np.all(hmap >= 0.0)


def test_sky_track_to_healpix_handles_multiple_passes() -> None:
    track = SkyTrack(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        timegrid=make_test_timegrid(),
        altitude_deg=np.array([-1.0, 20.0, 25.0, -2.0, 30.0, 35.0, -3.0]),
        azimuth_deg=np.array([0.0, 10.0, 20.0, 30.0, 180.0, 190.0, 200.0]),
        range_km=np.array([1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0]),
    )

    hmap = track.to_healpix(nside=16, unique_per_pass=True)

    assert hmap.shape == (hp.nside2npix(16),)
    assert np.sum(hmap) >= 2.0


def test_sky_track_to_healpix_skips_empty_pass(monkeypatch) -> None:
    track = SkyTrack(
        orbit=make_test_orbit(),
        observatory=Observatory.get("MWA"),
        timegrid=make_test_timegrid(),
        altitude_deg=np.array([10.0]),
        azimuth_deg=np.array([20.0]),
        range_km=np.array([1000.0]),
    )

    empty_pass = SkyPass(
        orbit=track.orbit,
        observatory=track.observatory,
        timegrid=track.timegrid,
        start_index=0,
        stop_index=0,
        altitude_deg=np.array([]),
        azimuth_deg=np.array([]),
        range_km=np.array([]),
    )

    # Force passes() to return an empty pass
    monkeypatch.setattr(SkyTrack, "passes", lambda self: [empty_pass])

    hmap = track.to_healpix(nside=8)

    # Nothing should be added
    assert np.all(hmap == 0.0)
