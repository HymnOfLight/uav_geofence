"""Planar world frame <-> geographic coordinate conversion.

The experiment world frame is x = east, y = north (meters) around a home
point. The same convention is used by the log loaders (NED -> world), the
SITL recorder, the fence/mission generator and the headless mission driver,
so data recorded against generated fences lines up with the experiment
geometry without manual offsets.
"""

from __future__ import annotations

import math

# PX4 SITL default home (Zurich Irchel); ArduPilot SITL accepts any home.
DEFAULT_HOME = (47.397742, 8.545594)
EARTH_M_PER_DEG = 111320.0


def xy_to_latlon(x_east: float, y_north: float, home_lat: float, home_lon: float) -> tuple[float, float]:
    lat = home_lat + y_north / EARTH_M_PER_DEG
    lon = home_lon + x_east / (EARTH_M_PER_DEG * math.cos(math.radians(home_lat)))
    return lat, lon


def latlon_to_xy(lat: float, lon: float, home_lat: float, home_lon: float) -> tuple[float, float]:
    x = (lon - home_lon) * EARTH_M_PER_DEG * math.cos(math.radians(home_lat))
    y = (lat - home_lat) * EARTH_M_PER_DEG
    return x, y


def box_polygon_latlon(
    box: tuple[float, float, float, float],
    pad: float,
    home: tuple[float, float],
) -> list[list[float]]:
    """Corners of the (padded) forbidden box as [lat, lon] pairs, CCW."""
    xmin, xmax, ymin, ymax = box
    corners_xy = [
        (xmin - pad, ymin - pad),
        (xmax + pad, ymin - pad),
        (xmax + pad, ymax + pad),
        (xmin - pad, ymax + pad),
    ]
    return [list(xy_to_latlon(x, y, *home)) for x, y in corners_xy]
