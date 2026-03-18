"""Satellite track calculations for ground and observatory sky coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import healpy as hp
import numpy as np
from skyfield.api import wgs84

from skyweaver import Observatory, OrbitSpec, TimeGrid


@dataclass(frozen=True, slots=True)
class GroundTrack:
    """Sub-satellite ground track sampled on a time grid."""

    orbit: OrbitSpec
    timegrid: TimeGrid
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray

    @property
    def n_times(self) -> int:
        """Return the number of sampled time points."""
        return len(self.latitude_deg)

    def summary(self) -> str:
        """Return a compact human-readable summary."""
        return f"GroundTrack(orbit={self.orbit.name!r}, n_times={self.n_times})"


@dataclass(frozen=True, slots=True)
class SkyPass:
    """Single contiguous above-horizon satellite pass."""

    orbit: OrbitSpec
    observatory: Observatory
    timegrid: TimeGrid
    start_index: int
    stop_index: int
    altitude_deg: np.ndarray
    azimuth_deg: np.ndarray
    range_km: np.ndarray

    @property
    def n_times(self) -> int:
        """Return the number of samples in the pass."""
        return len(self.altitude_deg)

    @property
    def max_altitude_deg(self) -> float:
        """Return the maximum altitude during the pass."""
        return float(np.max(self.altitude_deg))


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

    def passes(self) -> list["SkyPass"]:
        """Split the sky track into contiguous above-horizon passes."""
        visible = self.visible

        if not np.any(visible):
            return []

        visible_i = visible.astype(int)
        changes = np.diff(visible_i)

        starts = np.where(changes == 1)[0] + 1
        stops = np.where(changes == -1)[0] + 1

        if visible[0]:
            starts = np.r_[0, starts]

        if visible[-1]:
            stops = np.r_[stops, len(visible)]

        passes: list[SkyPass] = []
        for start, stop in zip(starts, stops, strict=True):
            passes.append(
                SkyPass(
                    orbit=self.orbit,
                    observatory=self.observatory,
                    timegrid=self.timegrid,
                    start_index=int(start),
                    stop_index=int(stop),
                    altitude_deg=self.altitude_deg[start:stop],
                    azimuth_deg=self.azimuth_deg[start:stop],
                    range_km=self.range_km[start:stop],
                )
            )

        return passes

    def to_healpix(
        self,
        nside: int,
        *,
        unique_per_pass: bool = True,
    ) -> np.ndarray:
        """Convert the sky track to a HEALPix map in local alt-az coordinates.

        Parameters
        ----------
        nside
            HEALPix nside parameter.
        unique_per_pass
            If True, each pixel is counted at most once per pass. If False,
            every visible sample contributes to the map.

        Returns
        -------
        np.ndarray
            HEALPix map of pass/sample counts.
        """
        npix = hp.nside2npix(nside)
        healpix_map = np.zeros(npix, dtype=float)

        for sat_pass in self.passes():
            if sat_pass.n_times == 0:
                continue

            theta = np.deg2rad(90.0 - sat_pass.altitude_deg)
            phi = np.deg2rad(sat_pass.azimuth_deg)

            pixels = hp.ang2pix(nside, theta, phi)

            if unique_per_pass:
                pixels = np.unique(pixels)

            np.add.at(healpix_map, pixels, 1.0)

        return healpix_map

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

    return GroundTrack(
        orbit=orbit,
        timegrid=timegrid,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
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
