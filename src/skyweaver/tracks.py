"""Satellite track calculations for ground and observatory sky coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import healpy as hp
import numpy as np
from skyfield.api import load, wgs84

from skyweaver.observatories import Observatory
from skyweaver.orbits import OrbitSpec
from skyweaver.timegrid import TimeGrid


def _sample_interval_datetimes(
    start_utc: datetime,
    stop_utc: datetime,
    cadence_s: float,
) -> list[datetime]:
    """Sample a closed UTC interval at a fixed cadence, including the end time."""
    if cadence_s <= 0.0:
        raise ValueError("cadence_s must be positive")

    duration_s = (stop_utc - start_utc).total_seconds()
    if duration_s < 0.0:
        raise ValueError("stop_utc must be greater than or equal to start_utc")

    offsets_s = np.arange(0.0, duration_s, cadence_s, dtype=float)
    datetimes = [start_utc + timedelta(seconds=float(offset_s)) for offset_s in offsets_s]

    if len(datetimes) == 0 or datetimes[-1] != stop_utc:
        datetimes.append(stop_utc)

    return datetimes


def _pair_visibility_events(
    event_times_utc: list[datetime],
    event_codes: np.ndarray,
    start_utc: datetime,
    stop_utc: datetime,
) -> tuple["PassInterval", ...]:
    """Pair Skyfield rise/set events into visibility intervals.

    Skyfield event codes are:
    - 0: rise
    - 1: culmination
    - 2: set

    If the search interval begins with the satellite already above the horizon,
    the interval start is treated as an effective rise. Likewise, if the search
    interval ends while the satellite is still above the horizon, the interval
    end is treated as an effective set.
    """
    mask = event_codes != 1
    filtered_times = [t for t, keep in zip(event_times_utc, mask, strict=True) if keep]
    filtered_codes = event_codes[mask]

    if len(filtered_codes) == 0:
        return ()

    if filtered_codes[0] == 2:
        filtered_times = [start_utc, *filtered_times]
        filtered_codes = np.r_[0, filtered_codes]

    if filtered_codes[-1] == 0:
        filtered_times = [*filtered_times, stop_utc]
        filtered_codes = np.r_[filtered_codes, 2]

    rise_times = [t for t, code in zip(filtered_times, filtered_codes, strict=True) if code == 0]
    set_times = [t for t, code in zip(filtered_times, filtered_codes, strict=True) if code == 2]

    if len(rise_times) != len(set_times):
        raise RuntimeError("Mismatched number of rise and set events")

    return tuple(
        PassInterval(start_utc=rise_time, stop_utc=set_time) for rise_time, set_time in zip(rise_times, set_times, strict=True)
    )


@dataclass(frozen=True, slots=True)
class PassInterval:
    """UTC bounds for a single above-horizon pass."""

    start_utc: datetime
    stop_utc: datetime

    @property
    def duration_s(self) -> float:
        """Return the interval duration in seconds."""
        return (self.stop_utc - self.start_utc).total_seconds()

    def datetimes(self, cadence_s: float) -> list[datetime]:
        """Return sampled UTC datetimes over the interval."""
        return _sample_interval_datetimes(
            self.start_utc,
            self.stop_utc,
            cadence_s,
        )


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
    interval: PassInterval
    cadence_s: float | None
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

    @property
    def start_utc(self) -> datetime:
        """Return the UTC start time of the pass."""
        return self.interval.start_utc

    @property
    def stop_utc(self) -> datetime:
        """Return the UTC stop time of the pass."""
        return self.interval.stop_utc

    def datetimes(self) -> list[datetime]:
        """Return UTC datetimes for samples in this pass."""
        if self.cadence_s is None:
            raise ValueError("This pass does not have a regular stored cadence")
        return self.interval.datetimes(self.cadence_s)


@dataclass(frozen=True, slots=True)
class SkyTrack:
    """Satellite sky track sampled at an observatory."""

    orbit: OrbitSpec
    observatory: Observatory
    altitude_deg: np.ndarray
    azimuth_deg: np.ndarray
    range_km: np.ndarray
    cadence_s: float | None = None
    timegrid: TimeGrid | None = None
    pass_intervals: tuple[PassInterval, ...] = ()
    pass_sample_counts: np.ndarray | None = None

    @property
    def n_times(self) -> int:
        """Return the number of sampled time points."""
        return len(self.altitude_deg)

    @property
    def visible(self) -> np.ndarray:
        """Return boolean mask where satellite is above the horizon."""
        return self.altitude_deg > 0.0

    def datetimes(self) -> list[datetime]:
        """Return UTC datetimes for the stored samples."""
        if self.timegrid is not None:
            return self.timegrid.datetimes()

        if self.cadence_s is None:
            raise ValueError("This SkyTrack does not have a regular stored cadence")

        datetimes: list[datetime] = []
        for interval in self.pass_intervals:
            datetimes.extend(interval.datetimes(self.cadence_s))
        return datetimes

    def passes(self) -> list["SkyPass"]:
        """Return contiguous above-horizon passes."""
        if self.pass_sample_counts is not None and len(self.pass_intervals) > 0:
            edges = np.r_[0, np.cumsum(self.pass_sample_counts)]
            passes: list[SkyPass] = []

            for i, interval in enumerate(self.pass_intervals):
                start = int(edges[i])
                stop = int(edges[i + 1])

                passes.append(
                    SkyPass(
                        orbit=self.orbit,
                        observatory=self.observatory,
                        interval=interval,
                        cadence_s=self.cadence_s,
                        start_index=start,
                        stop_index=stop,
                        altitude_deg=self.altitude_deg[start:stop],
                        azimuth_deg=self.azimuth_deg[start:stop],
                        range_km=self.range_km[start:stop],
                    )
                )

            return passes

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

        if self.timegrid is None:
            raise RuntimeError("Cannot derive coarse pass metadata without a TimeGrid")

        coarse_times = self.timegrid.datetimes()
        passes: list[SkyPass] = []

        for start, stop in zip(starts, stops, strict=True):
            interval = PassInterval(
                start_utc=coarse_times[int(start)],
                stop_utc=coarse_times[int(stop) - 1],
            )

            passes.append(
                SkyPass(
                    orbit=self.orbit,
                    observatory=self.observatory,
                    interval=interval,
                    cadence_s=self.cadence_s,
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
        Sub-satellite latitude and longitude as a function of time.
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
    *,
    pass_cadence_s: float | None = None,
) -> SkyTrack:
    """Compute the satellite sky track as seen from an observatory.

    Parameters
    ----------
    orbit
        Orbit specification to propagate.
    observatory
        Observatory from which to view the satellite.
    timegrid
        Time interval over which to search/evaluate the satellite.
    pass_cadence_s
        Optional finer cadence, in seconds, used to evaluate only the
        above-horizon pass windows identified by Skyfield event finding.

    Returns
    -------
    SkyTrack
        Altitude, azimuth, and range as a function of time.
    """
    satellite = orbit.to_earth_satellite()
    difference = satellite - observatory.skyfield_topos

    if pass_cadence_s is None:
        times = timegrid.skyfield()
        topocentric = difference.at(times)

        altitude, azimuth, distance = topocentric.altaz()

        return SkyTrack(
            orbit=orbit,
            observatory=observatory,
            altitude_deg=np.asarray(altitude.degrees, dtype=float),
            azimuth_deg=np.asarray(azimuth.degrees, dtype=float),
            range_km=np.asarray(distance.km, dtype=float),
            cadence_s=timegrid.cadence_s,
            timegrid=timegrid,
        )

    if pass_cadence_s <= 0.0:
        raise ValueError("pass_cadence_s must be positive")

    ts = load.timescale()
    t0 = ts.from_datetime(timegrid.start)
    t1 = ts.from_datetime(timegrid.stop)

    event_times, event_codes = satellite.find_events(
        observatory.skyfield_topos,
        t0,
        t1,
        altitude_degrees=0.0,
    )

    event_times_utc = list(event_times.utc_datetime())
    intervals = _pair_visibility_events(
        event_times_utc,
        event_codes,
        timegrid.start,
        timegrid.stop,
    )

    if len(intervals) == 0:
        return SkyTrack(
            orbit=orbit,
            observatory=observatory,
            altitude_deg=np.asarray([], dtype=float),
            azimuth_deg=np.asarray([], dtype=float),
            range_km=np.asarray([], dtype=float),
            cadence_s=pass_cadence_s,
            pass_intervals=(),
            pass_sample_counts=np.asarray([], dtype=int),
        )

    altitude_all: list[np.ndarray] = []
    azimuth_all: list[np.ndarray] = []
    range_all: list[np.ndarray] = []
    pass_sample_counts: list[int] = []

    for interval in intervals:
        pass_datetimes = interval.datetimes(pass_cadence_s)
        pass_times = ts.from_datetimes(pass_datetimes)

        topocentric = difference.at(pass_times)
        altitude, azimuth, distance = topocentric.altaz()

        altitude_deg = np.asarray(altitude.degrees, dtype=float)
        azimuth_deg = np.asarray(azimuth.degrees, dtype=float)
        range_km = np.asarray(distance.km, dtype=float)

        altitude_all.append(altitude_deg)
        azimuth_all.append(azimuth_deg)
        range_all.append(range_km)
        pass_sample_counts.append(len(pass_datetimes))

    return SkyTrack(
        orbit=orbit,
        observatory=observatory,
        altitude_deg=np.concatenate(altitude_all),
        azimuth_deg=np.concatenate(azimuth_all),
        range_km=np.concatenate(range_all),
        cadence_s=pass_cadence_s,
        pass_intervals=intervals,
        pass_sample_counts=np.asarray(pass_sample_counts, dtype=int),
    )
