from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from skyweaver.orbits import OrbitSpec, _ensure_utc, _wrap_angle_deg


def test_wrap_angle_deg_basic_cases() -> None:
    assert np.isclose(_wrap_angle_deg(0.0), 0.0)
    assert np.isclose(_wrap_angle_deg(360.0), 0.0)
    assert np.isclose(_wrap_angle_deg(-10.0), 350.0)
    assert np.isclose(_wrap_angle_deg(725.0), 5.0)


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


def test_invalid_altitude_raises() -> None:
    with pytest.raises(ValueError, match="altitude_km must be positive"):
        OrbitSpec.circular(
            name="bad_alt",
            epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
            altitude_km=0.0,
            inclination_deg=70.0,
        )


def test_invalid_eccentricity_raises() -> None:
    with pytest.raises(ValueError, match="eccentricity must satisfy 0 <= e < 1"):
        OrbitSpec(
            name="bad_e",
            epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
            altitude_km=550.0,
            inclination_deg=70.0,
            eccentricity=1.0,
        )


def test_invalid_inclination_raises() -> None:
    with pytest.raises(ValueError, match="inclination_deg must satisfy 0 <= i <= 180"):
        OrbitSpec.circular(
            name="bad_i",
            epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
            altitude_km=550.0,
            inclination_deg=181.0,
        )


def test_derived_orbital_properties_are_reasonable() -> None:
    orbit = OrbitSpec.circular(
        name="derived",
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        altitude_km=550.0,
        inclination_deg=70.0,
    )

    assert orbit.semi_major_axis_km > 6900.0
    assert orbit.mean_motion_rad_s > 0.0
    assert orbit.mean_motion_rad_min > 0.0
    assert 14.0 < orbit.mean_motion_rev_day < 16.0
    assert orbit.period_s > 0.0
    assert 90.0 < orbit.period_min < 100.0


def test_summary_contains_key_fields() -> None:
    orbit = OrbitSpec.circular(
        name="summary_test",
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        altitude_km=550.0,
        inclination_deg=70.0,
        raan_deg=30.0,
        phase_deg=120.0,
    )

    summary = orbit.summary()

    assert "summary_test" in summary
    assert "altitude_km=550.0" in summary
    assert "inclination_deg=70.000" in summary
    assert "raan_deg=30.000" in summary
    assert "mean_anomaly_deg=120.000" in summary
    assert "period_min=" in summary


def test_circular_orbit_defaults() -> None:
    orbit = OrbitSpec.circular(
        name="test",
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        altitude_km=550.0,
        inclination_deg=70.0,
        raan_deg=30.0,
        phase_deg=120.0,
    )

    assert orbit.eccentricity == 0.0
    assert orbit.arg_perigee_deg == 0.0
    assert orbit.mean_anomaly_deg == 120.0


def test_period_is_reasonable_for_leo() -> None:
    orbit = OrbitSpec.circular(
        name="test",
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        altitude_km=550.0,
        inclination_deg=70.0,
    )

    assert 90.0 < orbit.period_min < 100.0


def test_angle_wrapping() -> None:
    orbit = OrbitSpec.circular(
        name="test",
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        altitude_km=550.0,
        inclination_deg=70.0,
        raan_deg=390.0,
        phase_deg=-10.0,
    )

    assert np.isclose(orbit.raan_deg, 30.0)
    assert np.isclose(orbit.mean_anomaly_deg, 350.0)


def test_to_satrec_builds() -> None:
    orbit = OrbitSpec.circular(
        name="test",
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        altitude_km=550.0,
        inclination_deg=70.0,
    )

    satrec = orbit.to_satrec()
    assert satrec.satnum == 99999


def test_to_earth_satellite_builds() -> None:
    orbit = OrbitSpec.circular(
        name="my_sat",
        epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        altitude_km=550.0,
        inclination_deg=70.0,
    )

    satellite = orbit.to_earth_satellite()
    assert satellite.name == "my_sat"
