"""Ground-track plotting utilities."""

from __future__ import annotations

from collections.abc import Sequence

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from mpl_toolkits.basemap import Basemap

from skyweaver import GroundTrack, Observatory


def plot_ground_track(
    track: GroundTrack,
    ax: Axes | None = None,
    *,
    observatories: Sequence[Observatory] | None = None,
    show_labels: bool = True,
) -> Axes:
    """Plot a simple longitude-latitude ground track.

    Parameters
    ----------
    track
        Ground track to plot.
    ax
        Optional matplotlib axes. If not provided, a new figure and axes are
        created.
    observatories
        Optional sequence of observatories to mark on the map.
    show_labels
        If True, annotate observatory names next to their markers.

    Returns
    -------
    Axes
        Matplotlib axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    m = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
        resolution="c",
        ax=ax,
    )

    land_color = to_hex(cmr.pride(0.36))

    m.fillcontinents(color=land_color, lake_color=land_color, zorder=42)
    m.drawcountries(color="whitesmoke", zorder=42, linewidth=0.7)
    m.drawcoastlines(color="whitesmoke", zorder=42, linewidth=0.7)
    m.drawparallels(np.arange(-90.0, 91.0, 30.0), linewidth=0.5, alpha=0.7)
    m.drawmeridians(np.arange(-180.0, 181.0, 60.0), linewidth=0.5, alpha=0.7)

    ax.plot(
        track.longitude_deg,
        track.latitude_deg,
        lw=0.1,
        alpha=0.9,
        color=cmr.pride(0.77),
    )

    if observatories is not None:
        for obs in observatories:
            x, y = m(obs.longitude_deg, obs.latitude_deg)
            x = float(str(x))
            y = float(str(y))

            ax.scatter(
                x,
                y,
                s=121,
                marker=r"$⬢$",
                color=cmr.pride(0.1),
                edgecolors="whitesmoke",
                linewidths=0.7,
                zorder=49,
            )

            if show_labels:
                ax.annotate(
                    obs.name,
                    xy=(x, y),
                    xytext=(7, 4),
                    textcoords="offset points",
                    fontsize=9,
                    ha="left",
                    va="bottom",
                    zorder=49,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="whitesmoke",
                        edgecolor="#222222",
                        linewidth=0.4,
                        alpha=0.9,
                    ),
                )

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"Ground track: {track.orbit.name}")
    ax.set_xticks(np.arange(-120, 121, 60))
    ax.set_yticks(np.arange(-60, 61, 30))
    ax.grid(True)

    return ax
