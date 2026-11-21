"""Geo transforms: lon/lat (or any CRS) <-> drawing frame (metres).

Site-model frame: local tangent plane at the Rhino EarthAnchorPoint
(azimuthal equidistant, +Y true north), millimetres.
Drawing frame: rotated/translated per project.yaml site block, metres,
Z = 0 at Level 1.
"""
from __future__ import annotations

import math

import numpy as np
from pyproj import CRS, Transformer


class Geo:
    def __init__(self, spec):
        s = spec["project"]["site"]
        self.lat, self.lon = s["anchor_lat"], s["anchor_lon"]
        self.th = math.radians(s["plan_rotation_deg"])
        self.o = np.array(s["plan_origin_mm"]) / 1000.0
        self.z0 = s["level1_elevation_mm"] / 1000.0
        self.local = CRS.from_proj4(f"+proj=aeqd +lat_0={self.lat} +lon_0={self.lon} +datum=WGS84 +units=m")
        self._tx = {}

    def _t(self, src, dst):
        k = (str(src), str(dst))
        if k not in self._tx:
            self._tx[k] = Transformer.from_crs(src, dst, always_xy=True)
        return self._tx[k]

    # site-model (m) <-> drawing frame (m)
    def site_to_plan(self, xy):
        xy = np.asarray(xy, float) - self.o
        c, s = math.cos(self.th), math.sin(self.th)
        return np.c_[c * xy[..., 0] + s * xy[..., 1], -s * xy[..., 0] + c * xy[..., 1]]

    def plan_to_site(self, xy):
        xy = np.asarray(xy, float)
        c, s = math.cos(self.th), math.sin(self.th)
        return np.c_[c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]] + self.o

    def lonlat_to_plan(self, lon, lat):
        x, y = self._t("EPSG:4326", self.local).transform(np.asarray(lon), np.asarray(lat))
        return self.site_to_plan(np.c_[x, y])

    def plan_to_lonlat(self, xy):
        s = self.plan_to_site(xy)
        lon, lat = self._t(self.local, "EPSG:4326").transform(s[:, 0], s[:, 1])
        return np.c_[lon, lat]

    def crs_to_plan(self, crs, x, y):
        lx, ly = self._t(crs, self.local).transform(np.asarray(x), np.asarray(y))
        return self.site_to_plan(np.c_[lx, ly])

    def plan_to_crs(self, crs, xy):
        s = self.plan_to_site(xy)
        return np.c_[self._t(self.local, crs).transform(s[:, 0], s[:, 1])]

    def plan_north(self):
        """Unit vector of true north in the drawing frame."""
        return self.site_to_plan(self.o + np.array([0.0, 1.0]))[0] - self.site_to_plan(self.o)[0]
