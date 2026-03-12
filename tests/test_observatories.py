from __future__ import annotations

import numpy as np
import pytest

from skyweaver.observatories import Observatory


@pytest.fixture
def restore_registry() -> None:
    """Preserve and restore the observatory registry for tests that mutate it."""
    original = dict(Observatory._registry)
    try:
        yield
    finally:
        Observatory._registry.clear()
        Observatory._registry.update(original)


def test_builtin_registry_contains_expected_names() -> None:
    expected = {
        "ALBATROS",
        "CHORD",
        "HERA",
        "HIRAX",
        "LOFAR",
        "MWA",
        "NENUFAR",
        "OVRO_LWA",
        "SKA_LOW",
    }

    assert set(Observatory.names()) == expected


def test_get_returns_builtin_observatory_case_insensitive() -> None:
    mwa_upper = Observatory.get("MWA")
    mwa_lower = Observatory.get("mwa")

    assert mwa_upper.name == "MWA"
    assert mwa_lower.name == "MWA"
    assert mwa_upper == mwa_lower


def test_all_returns_copy_of_registry() -> None:
    registry_copy = Observatory.all()

    assert isinstance(registry_copy, dict)
    assert "MWA" in registry_copy

    registry_copy.pop("MWA")
    assert "MWA" in Observatory.all()


def test_get_unknown_observatory_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        Observatory.get("NOT_A_REAL_SITE")


def test_skyfield_topos_has_expected_mwa_coordinates() -> None:
    mwa = Observatory.get("MWA")
    topos = mwa.skyfield_topos

    assert np.isclose(topos.latitude.degrees, -26.703319)
    assert np.isclose(topos.longitude.degrees, 116.670815)
    assert np.isclose(topos.elevation.m, 377.83)


def test_direct_instantiation_registers_observatory(restore_registry: None) -> None:
    site = Observatory(
        name="TESTSITE",
        latitude_deg=12.34,
        longitude_deg=-56.78,
        elevation_m=90.0,
    )

    fetched = Observatory.get("TESTSITE")

    assert fetched == site
    assert fetched.name == "TESTSITE"
    assert np.isclose(fetched.latitude_deg, 12.34)
    assert np.isclose(fetched.longitude_deg, -56.78)
    assert np.isclose(fetched.elevation_m, 90.0)


def test_registry_overwrites_same_name_with_latest_instance(
    restore_registry: None,
) -> None:
    first = Observatory(
        name="DUPLICATE",
        latitude_deg=1.0,
        longitude_deg=2.0,
        elevation_m=3.0,
    )
    second = Observatory(
        name="DUPLICATE",
        latitude_deg=4.0,
        longitude_deg=5.0,
        elevation_m=6.0,
    )

    fetched = Observatory.get("DUPLICATE")

    assert first != second
    assert fetched == second
    assert np.isclose(fetched.latitude_deg, 4.0)
    assert np.isclose(fetched.longitude_deg, 5.0)
    assert np.isclose(fetched.elevation_m, 6.0)


@pytest.mark.parametrize(
    ("factory_name", "expected_name", "lat", "lon", "elev"),
    [
        ("mwa", "MWA", -26.703319, 116.670815, 377.83),
        ("lofar", "LOFAR", 52.9088, 6.8677, 15.0),
        ("hera", "HERA", -30.7215, 21.4283, 1050.0),
        ("ska_low", "SKA_LOW", -26.824722, 116.764448, 365.0),
        ("nenufar", "NENUFAR", 47.3765, 2.1924, 130.0),
        ("ovro_lwa", "OVRO_LWA", 37.2398, -118.2817, 1183.0),
        ("albatros", "ALBATROS", 79.417183, -90.76735, 189.0),
        ("chord", "CHORD", 49.3207, -119.6200, 545.0),
        ("hirax", "HIRAX", -30.721, 21.411, 1080.0),
    ],
)
def test_builtin_constructors(
    factory_name: str,
    expected_name: str,
    lat: float,
    lon: float,
    elev: float,
) -> None:
    constructor = getattr(Observatory, factory_name)
    site = constructor()

    assert site.name == expected_name
    assert np.isclose(site.latitude_deg, lat)
    assert np.isclose(site.longitude_deg, lon)
    assert np.isclose(site.elevation_m, elev)
