"""Observatory site definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict

from skyfield.api import wgs84
from skyfield.toposlib import GeographicPosition


@dataclass(frozen=True, slots=True)
class Observatory:
    """Geographic location of an observing site."""

    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float = 0.0

    _registry: ClassVar[Dict[str, "Observatory"]] = {}

    def __post_init__(self) -> None:
        """Register observatory instance."""
        self._registry[self.name.upper()] = self

    @property
    def skyfield_topos(self) -> GeographicPosition:
        """Return a Skyfield WGS84 Topos object."""
        return wgs84.latlon(
            self.latitude_deg,
            self.longitude_deg,
            elevation_m=self.elevation_m,
        )

    # ---- Registry helpers ----

    @classmethod
    def get(cls, name: str) -> "Observatory":
        """Retrieve an observatory by name."""
        return cls._registry[name.upper()]

    @classmethod
    def all(cls) -> dict[str, "Observatory"]:
        """Return dictionary of registered observatories."""
        return dict(cls._registry)

    @classmethod
    def names(cls) -> list[str]:
        """Return list of registered observatory names."""
        return sorted(cls._registry)

    # ---- Observatory constructors ----

    @classmethod
    def mwa(cls) -> "Observatory":
        return cls("MWA", -26.703319, 116.670815, 377.83)

    @classmethod
    def lofar(cls) -> "Observatory":
        return cls("LOFAR", 52.9088, 6.8677, 15.0)

    @classmethod
    def hera(cls) -> "Observatory":
        return cls("HERA", -30.7215, 21.4283, 1050.0)

    @classmethod
    def ska_low(cls) -> "Observatory":
        return cls("SKA_LOW", -26.824722, 116.764448, 365.0)

    @classmethod
    def nenufar(cls) -> "Observatory":
        return cls("NENUFAR", 47.3765, 2.1924, 130.0)

    @classmethod
    def ovro_lwa(cls) -> "Observatory":
        return cls("OVRO_LWA", 37.2398, -118.2817, 1183.0)

    @classmethod
    def albatros(cls) -> "Observatory":
        return cls("ALBATROS", 79.417183, -90.76735, 189.0)

    @classmethod
    def chord(cls) -> "Observatory":
        return cls("CHORD", 49.3207, -119.6200, 545.0)

    @classmethod
    def hirax(cls) -> "Observatory":
        return cls("HIRAX", -30.721, 21.411, 1080.0)


# ---- Initialize registry with built-ins ----

Observatory.mwa()
Observatory.lofar()
Observatory.hera()
Observatory.ska_low()
Observatory.nenufar()
Observatory.ovro_lwa()
Observatory.albatros()
Observatory.chord()
Observatory.hirax()
