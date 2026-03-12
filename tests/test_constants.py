from __future__ import annotations

import numpy as np

from skyweaver.constants import (
    EARTH_EQUATORIAL_RADIUS_KM,
    EARTH_MU_KM3_S2,
    MINUTES_PER_DAY,
    SECONDS_PER_DAY,
    SECONDS_PER_MINUTE,
    SGP4_EPOCH_OFFSET_JD,
)


def test_earth_radius_is_reasonable() -> None:
    assert np.isclose(EARTH_EQUATORIAL_RADIUS_KM, 6378.137)


def test_earth_mu_is_reasonable() -> None:
    assert np.isclose(EARTH_MU_KM3_S2, 398600.4418)


def test_day_constants_are_consistent() -> None:
    assert SECONDS_PER_MINUTE == 60.0
    assert MINUTES_PER_DAY == 1440.0
    assert SECONDS_PER_DAY == 86400.0
    assert np.isclose(SECONDS_PER_DAY / MINUTES_PER_DAY, 60.0)


def test_sgp4_epoch_offset_is_reasonable() -> None:
    assert np.isclose(SGP4_EPOCH_OFFSET_JD, 2433281.5)
