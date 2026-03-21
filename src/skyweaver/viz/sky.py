"""Sky-track plotting utilities."""

from __future__ import annotations

from typing import cast

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Colormap
from matplotlib.projections.polar import PolarAxes

from skyweaver.tracks import SkyPass, SkyTrack


def _plot_single_pass(
    sat_pass: SkyPass,
    ax: PolarAxes,
    *,
    color: str | tuple[float, float, float, float] | None = None,
    alpha: float = 1.0,
    lw: float = 1.5,
) -> None:
    """Plot a single above-horizon pass on polar axes."""
    if sat_pass.n_times == 0:
        return

    theta = np.deg2rad(sat_pass.azimuth_deg)
    radius = sat_pass.altitude_deg

    ax.plot(theta, radius, lw=lw, alpha=alpha, color=color)


def plot_sky_track(
    track: SkyTrack,
    ax: PolarAxes | None = None,
    *,
    cmap: str | Colormap | None = None,
    color: str | None = None,
    alpha: float = 1.0,
    lw: float = 1.5,
) -> PolarAxes:
    """Plot a satellite sky track in polar alt-az coordinates."""
    if ax is None:
        _, ax0 = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
        ax = cast(PolarAxes, ax0)

    passes = track.passes()

    if cmap is None:
        pass_colors = [color] * len(passes)
    else:
        cmap_obj = colormaps[cmap] if isinstance(cmap, str) else cmap

        n_passes = len(passes)
        if n_passes == 0:
            pass_colors = []
        elif n_passes == 1:
            pass_colors = [cmap_obj(0.5)]
        else:
            pass_colors = list(cmap_obj(np.linspace(0.0, 1.0, n_passes)))

    for sat_pass, pass_color in zip(passes, pass_colors, strict=True):
        _plot_single_pass(
            sat_pass,
            ax,
            color=pass_color,
            alpha=alpha,
            lw=lw,
        )

    ax.set_title(f"Sky track: {track.orbit.name} at {track.observatory.name}")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(90.0, 0.0)
    ax.set_rgrids([30.0, 60.0, 90.0])
    ax.set_rlabel_position(67.5)
    ax.grid(True)

    return ax


def plot_sky_track_healpix(
    track: SkyTrack,
    *,
    nside: int = 32,
    unique_per_pass: bool = True,
    cmap=None,
    title: str | None = None,
    unit: str = "counts",
    half_sky: bool = True,
    rot: list[float] | tuple[float, float, float] = (0, 90, 180),
    graticule: bool = True,
    **kwargs,
) -> np.ndarray:
    """Plot a HEALPix map of a sky track in local alt-az coordinates.

    Parameters
    ----------
    track
        Sky track to bin into HEALPix pixels.
    nside
        HEALPix nside parameter.
    unique_per_pass
        If True, count each pixel at most once per pass.
    cmap
        Optional colormap passed to healpy.
    title
        Plot title. If None, a default title is used.
    unit
        Colorbar/unit label passed to healpy.
    half_sky
        Whether to use a half-sky orthographic projection.
    rot
        Rotation passed to ``healpy.orthview``.
    graticule
        If True, overlay a graticule and cardinal directions.
    **kwargs
        Additional keyword arguments passed to ``healpy.orthview``.

    Returns
    -------
    np.ndarray
        The HEALPix map that was plotted.
    """
    healpix_map = track.to_healpix(
        nside=nside,
        unique_per_pass=unique_per_pass,
    )

    if title is None:
        title = f"Sky HEALPix map: {track.orbit.name} at {track.observatory.name}"

    hp.orthview(
        healpix_map,
        rot=rot,
        half_sky=half_sky,
        cmap=cmap,
        unit=unit,
        title=title,
        **kwargs,
    )

    if graticule:
        hp.visufunc.graticule(15, 30, ls=":", color="whitesmoke", alpha=0.5)

        hp.visufunc.projtext(
            0,
            21,
            "N",
            lonlat=True,
            ha="center",
            va="center",
            fontsize=14,
            color="w",
        )
        hp.visufunc.projtext(
            90,
            21,
            "E",
            lonlat=True,
            ha="center",
            va="center",
            fontsize=14,
            color="w",
        )
        hp.visufunc.projtext(
            180,
            21,
            "S",
            lonlat=True,
            ha="center",
            va="center",
            fontsize=14,
            color="w",
        )
        hp.visufunc.projtext(
            270,
            21,
            "W",
            lonlat=True,
            ha="center",
            va="center",
            fontsize=14,
            color="w",
        )

    return healpix_map
