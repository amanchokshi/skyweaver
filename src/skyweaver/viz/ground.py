"""Ground-track plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from skyweaver.tracks import GroundTrack


def plot_ground_track(
    track: GroundTrack,
    ax: Axes | None = None,
) -> Axes:
    """Plot a simple longitude-latitude ground track.

    Parameters
    ----------
    track
        Ground track to plot.
    ax
        Optional matplotlib axes. If not provided, a new figure and axes are
        created.

    Returns
    -------
    Axes
        Matplotlib axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(track.longitude_deg, track.latitude_deg, lw=1.5)

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"Ground track: {track.orbit.name}")
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.grid(True)

    return ax
