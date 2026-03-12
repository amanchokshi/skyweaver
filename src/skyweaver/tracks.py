"""Satellite track calculations for ground and observatory sky coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skyfield.api import wgs84

from skyweaver.observatories import Observatory
from skyweaver.orbits import OrbitSpec
from skyweaver.timegrid import TimeGrid


@dataclass(frozen=True, slots=True)
class GroundTrack:
    """Sub-satellite ground track sampled on a time grid."""

    orbit: OrbitSpec
    timegrid: TimeGrid
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    elevation_m: np.ndarray

    @property
    def n_times(self) -> int:
        """Return the number of sampled time points."""
        return len(self.latitude_deg)

    def summary(self) -> str:
        """Return a compact human-readable summary."""
        return f"GroundTrack(orbit={self.orbit.name!r}, n_times={self.n_times})"


@dataclass(frozen=True, slots=True)
class SkyTrack:
    """Satellite sky track sampled at an observatory on a time grid."""

    orbit: OrbitSpec
    observatory: Observatory
    timegrid: TimeGrid
    altitude_deg: np.ndarray
    azimuth_deg: np.ndarray
    range_km: np.ndarray

    @property
    def n_times(self) -> int:
        """Return the number of sampled time points."""
        return len(self.altitude_deg)

    @property
    def visible(self) -> np.ndarray:
        """Return boolean mask where satellite is above the horizon."""
        return self.altitude_deg > 0.0

    def summary(self) -> str:
        """Return a compact human-readable summary."""
        return f"SkyTrack(orbit={self.orbit.name!r}, observatory={self.observatory.name!r}, n_times={self.n_times})"


def ground_track(orbit: OrbitSpec, timegrid: TimeGrid) -> GroundTrack:
    """Compute the sub-satellite ground track for an orbit.

    Parameters
    ----------
    orbit
        Orbit specification to propagate.
    timegrid
        Time grid on which to evaluate the orbit.

    Returns
    -------
    GroundTrack
        Sub-satellite latitude, longitude, and elevation as a function of time.
    """
    satellite = orbit.to_earth_satellite()
    times = timegrid.skyfield()

    geocentric = satellite.at(times)
    subpoint = wgs84.subpoint_of(geocentric)

    latitude_deg = np.asarray(subpoint.latitude.degrees, dtype=float)
    longitude_deg = np.asarray(subpoint.longitude.degrees, dtype=float)
    elevation_m = np.broadcast_to(
        np.asarray(subpoint.elevation.m, dtype=float),
        latitude_deg.shape,
    ).astype(float, copy=False)

    return GroundTrack(
        orbit=orbit,
        timegrid=timegrid,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        elevation_m=elevation_m,
    )


def sky_track(
    orbit: OrbitSpec,
    observatory: Observatory,
    timegrid: TimeGrid,
) -> SkyTrack:
    """Compute the satellite sky track as seen from an observatory.

    Parameters
    ----------
    orbit
        Orbit specification to propagate.
    observatory
        Observatory from which to view the satellite.
    timegrid
        Time grid on which to evaluate the orbit.

    Returns
    -------
    SkyTrack
        Altitude, azimuth, and range as a function of time.
    """
    satellite = orbit.to_earth_satellite()
    times = timegrid.skyfield()

    difference = satellite - observatory.skyfield_topos
    topocentric = difference.at(times)

    altitude, azimuth, distance = topocentric.altaz()

    return SkyTrack(
        orbit=orbit,
        observatory=observatory,
        timegrid=timegrid,
        altitude_deg=np.asarray(altitude.degrees, dtype=float),
        azimuth_deg=np.asarray(azimuth.degrees, dtype=float),
        range_km=np.asarray(distance.km, dtype=float),
    )
