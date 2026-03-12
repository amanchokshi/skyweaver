"""skyweaver: simulate satellite orbits and how they sample observatory skies."""

from importlib.metadata import PackageNotFoundError, version

from skyweaver.observatories import Observatory
from skyweaver.orbits import OrbitSpec
from skyweaver.timegrid import TimeGrid
from skyweaver.tracks import GroundTrack, SkyTrack, ground_track, sky_track

try:
    __version__ = version("skyweaver")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "GroundTrack",
    "Observatory",
    "OrbitSpec",
    "SkyTrack",
    "TimeGrid",
    "ground_track",
    "sky_track",
]
