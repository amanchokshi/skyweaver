"""Sky-track plotting utilities."""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.projections.polar import PolarAxes

from skyweaver.tracks import SkyTrack


def plot_sky_track(
    track: SkyTrack,
    ax: PolarAxes | None = None,
) -> PolarAxes:
    """Plot a satellite sky track in polar alt-az coordinates.

    Parameters
    ----------
    track
        Sky track to plot.
    ax
        Optional polar matplotlib axes. If not provided, a new figure and polar
        axes are created.

    Returns
    -------
    PolarAxes
        Matplotlib polar axes containing the plot.
    """
    if ax is None:
        _, ax0 = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
        ax = cast(PolarAxes, ax0)

    theta = np.deg2rad(track.azimuth_deg)
    radius = track.altitude_deg

    ax.plot(theta, radius, lw=1.5)

    ax.set_title(f"Sky track: {track.orbit.name} at {track.observatory.name}")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(90, 0)
    ax.set_rgrids([15, 30, 45, 60, 75, 90])
    ax.set_rlabel_position(67.5)
    ax.grid(True)

    return ax
