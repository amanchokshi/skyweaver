"""Physical constants used across skyweaver."""

from __future__ import annotations

EARTH_EQUATORIAL_RADIUS_KM = 6378.137
EARTH_MU_KM3_S2 = 398600.4418

SECONDS_PER_MINUTE = 60.0
SECONDS_PER_DAY = 86400.0
MINUTES_PER_DAY = 1440.0

# SGP4 expects its epoch as:
# "days since 1949 December 31 00:00 UT"
SGP4_EPOCH_OFFSET_JD = 2433281.5
