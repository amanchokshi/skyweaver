"""Orbit specifications and conversion to SGP4/Skyfield objects.

This module defines a lightweight, physical orbit description for artificial
Earth satellites and provides conversion utilities for building `sgp4` and
Skyfield satellite objects.

The initial design target is circular or near-circular LEO orbits for
simulation and coverage studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sgp4.api import WGS72, Satrec
from sgp4.conveniences import check_satrec, jday_datetime
from skyfield.api import load
from skyfield.sgp4lib import EarthSatellite

from skyweaver.constants import (
    EARTH_EQUATORIAL_RADIUS_KM,
    EARTH_MU_KM3_S2,
    SECONDS_PER_DAY,
    SECONDS_PER_MINUTE,
    SGP4_EPOCH_OFFSET_JD,
)

_TWO_PI = 2.0 * np.pi


def _wrap_angle_deg(angle_deg: float) -> float:
    """Wrap an angle in degrees to the half-open interval [0, 360)."""
    return angle_deg % 360.0


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
class OrbitSpec:
    """A simple physical specification for an artificial Earth orbit.

    Parameters
    ----------
    name
        Human-readable orbit name.
    epoch
        Epoch of the element set. Naive datetimes are interpreted as UTC.
    altitude_km
        Mean altitude above the WGS84 equatorial radius, in kilometers.
    inclination_deg
        Inclination in degrees.
    raan_deg
        Right ascension of the ascending node, in degrees.
    mean_anomaly_deg
        Mean anomaly at epoch, in degrees.
    eccentricity
        Orbital eccentricity. For the initial `skyweaver` use case, this will
        generally be 0.0 or very small.
    arg_perigee_deg
        Argument of perigee in degrees. For circular orbits this is formally
        arbitrary, but SGP4 still requires a value.
    bstar
        Ballistic drag coefficient in inverse Earth radii. Leave at 0.0 for
        idealized artificial design studies.

    Notes
    -----
    This class stores a small, physically meaningful set of inputs that can
    later be converted into an SGP4 satellite record.
    """

    name: str
    epoch: datetime
    altitude_km: float
    inclination_deg: float
    raan_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    eccentricity: float = 0.0
    arg_perigee_deg: float = 0.0
    bstar: float = 0.0

    def __post_init__(self) -> None:
        """Validate fields and normalize periodic angles."""
        epoch_utc = _ensure_utc(self.epoch)
        object.__setattr__(self, "epoch", epoch_utc)

        if self.altitude_km <= 0.0:
            raise ValueError("altitude_km must be positive")

        if not (0.0 <= self.eccentricity < 1.0):
            raise ValueError("eccentricity must satisfy 0 <= e < 1")

        if not (0.0 <= self.inclination_deg <= 180.0):
            raise ValueError("inclination_deg must satisfy 0 <= i <= 180")

        object.__setattr__(self, "raan_deg", _wrap_angle_deg(self.raan_deg))
        object.__setattr__(
            self,
            "mean_anomaly_deg",
            _wrap_angle_deg(self.mean_anomaly_deg),
        )
        object.__setattr__(
            self,
            "arg_perigee_deg",
            _wrap_angle_deg(self.arg_perigee_deg),
        )

    @classmethod
    def circular(
        cls,
        *,
        name: str,
        epoch: datetime,
        altitude_km: float,
        inclination_deg: float,
        raan_deg: float = 0.0,
        phase_deg: float = 0.0,
        bstar: float = 0.0,
    ) -> "OrbitSpec":
        """Construct a circular orbit.

        Parameters
        ----------
        name
            Orbit name.
        epoch
            Epoch of the orbit. Naive datetimes are interpreted as UTC.
        altitude_km
            Altitude above Earth's equatorial radius.
        inclination_deg
            Inclination in degrees.
        raan_deg
            Right ascension of the ascending node in degrees.
        phase_deg
            Orbital phase, mapped here onto mean anomaly for circular orbits.
        bstar
            Optional drag term for SGP4.
        """
        return cls(
            name=name,
            epoch=epoch,
            altitude_km=altitude_km,
            inclination_deg=inclination_deg,
            raan_deg=raan_deg,
            mean_anomaly_deg=phase_deg,
            eccentricity=0.0,
            arg_perigee_deg=0.0,
            bstar=bstar,
        )

    @property
    def semi_major_axis_km(self) -> float:
        """Return the semi-major axis in kilometers."""
        return EARTH_EQUATORIAL_RADIUS_KM + self.altitude_km

    @property
    def mean_motion_rad_s(self) -> float:
        """Return the Keplerian mean motion in radians per second."""
        a_km = self.semi_major_axis_km
        return float(np.sqrt(EARTH_MU_KM3_S2 / (a_km**3)))

    @property
    def mean_motion_rad_min(self) -> float:
        """Return the mean motion in radians per minute."""
        return self.mean_motion_rad_s * SECONDS_PER_MINUTE

    @property
    def mean_motion_rev_day(self) -> float:
        """Return the mean motion in revolutions per day."""
        return self.mean_motion_rad_s * SECONDS_PER_DAY / _TWO_PI

    @property
    def period_s(self) -> float:
        """Return the orbital period in seconds."""
        return _TWO_PI / self.mean_motion_rad_s

    @property
    def period_min(self) -> float:
        """Return the orbital period in minutes."""
        return self.period_s / SECONDS_PER_MINUTE

    def to_satrec(self, satnum: int = 99999) -> Satrec:
        """Convert this orbit specification into an `sgp4.api.Satrec`.

        Parameters
        ----------
        satnum
            Satellite number to assign to the record.

        Returns
        -------
        Satrec
            Initialized SGP4 satellite record.
        """
        jd, fr = jday_datetime(self.epoch)
        sgp4_epoch_days = (jd + fr) - SGP4_EPOCH_OFFSET_JD

        satrec = Satrec()
        satrec.sgp4init(
            WGS72,
            "i",
            satnum,
            sgp4_epoch_days,
            self.bstar,
            0.0,
            0.0,
            self.eccentricity,
            np.deg2rad(self.arg_perigee_deg),
            np.deg2rad(self.inclination_deg),
            np.deg2rad(self.mean_anomaly_deg),
            self.mean_motion_rad_min,
            np.deg2rad(self.raan_deg),
        )
        check_satrec(satrec)
        return satrec

    def to_earth_satellite(self, satnum: int = 99999) -> EarthSatellite:
        """Wrap this orbit as a Skyfield `EarthSatellite`.

        Parameters
        ----------
        satnum
            Satellite number to assign to the underlying SGP4 record.

        Returns
        -------
        EarthSatellite
            Skyfield Earth satellite object.
        """
        ts = load.timescale()
        satrec = self.to_satrec(satnum=satnum)
        satellite = EarthSatellite.from_satrec(satrec, ts)
        satellite.name = self.name
        return satellite

    def summary(self) -> str:
        """Return a compact human-readable orbit summary."""
        return (
            f"OrbitSpec(name={self.name!r}, altitude_km={self.altitude_km:.1f}, "
            f"inclination_deg={self.inclination_deg:.3f}, "
            f"raan_deg={self.raan_deg:.3f}, "
            f"mean_anomaly_deg={self.mean_anomaly_deg:.3f}, "
            f"eccentricity={self.eccentricity:.6f}, "
            f"period_min={self.period_min:.2f})"
        )
