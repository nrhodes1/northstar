"""Load + cache site data in the drawing frame (metres, z rel. L1 FFE)."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from .geo import Geo

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"

FILES = {
    "buildings": "Boston_Buildings_with_Roof_Breaks.geojson",
    "sidewalks": "sidewalk_inventory.geojson",
    "streets": "boston_street_segments_sam_system.geojson",
    "parcels": "Parcels__2024_.geojson",
    "street_trees": "bprd_trees.csv",
}


@dataclass
class SiteData:
    geo: Geo
    R: float
    buildings: list      # [(Polygon, props)]
    sidewalks: list
    streets: list        # [(LineString, props)]
    parcels: list
    street_trees: list   # [(x, y, props)]
    dem: "Raster"
    ortho: "Ortho"
    dsm: "Raster"


class Raster:
    """Regular grid in the drawing frame. z[i, j] at (x0 + (j+.5)*res, y0 + (i+.5)*res)."""

    def __init__(self, z, x0, y0, res):
        self.z, self.x0, self.y0, self.res = z, x0, y0, res

    def sample(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        fx = (x - self.x0) / self.res - 0.5
        fy = (y - self.y0) / self.res - 0.5
        j0 = np.clip(np.floor(fx).astype(int), 0, self.z.shape[1] - 2)
        i0 = np.clip(np.floor(fy).astype(int), 0, self.z.shape[0] - 2)
        tx = np.clip(fx - j0, 0, 1)
        ty = np.clip(fy - i0, 0, 1)
        z = self.z
        return ((z[i0, j0] * (1 - tx) + z[i0, j0 + 1] * tx) * (1 - ty)
                + (z[i0 + 1, j0] * (1 - tx) + z[i0 + 1, j0 + 1] * tx) * ty)

    def xy(self):
        n_i, n_j = self.z.shape
        xs = self.x0 + (np.arange(n_j) + 0.5) * self.res
        ys = self.y0 + (np.arange(n_i) + 0.5) * self.res
        return np.meshgrid(xs, ys)


class Ortho:
    def __init__(self, img, top, H, res):
        self.img, self.top, self.H, self.res = img, top, H, res

    def rgb(self, x, y):
        j = np.clip(((np.asarray(x) + self.H) / self.res).astype(int), 0, self.img.shape[1] - 1)
        i = np.clip(((np.asarray(y) + self.H) / self.res).astype(int), 0, self.img.shape[0] - 1)
        return self.img[i, j]


def _cached_subset(name, geo, R, loader):
    sub = CACHE / f"sub_{name}_{int(R)}.json"
    if sub.exists():
        return json.loads(sub.read_text())
    feats = loader()
    sub.write_text(json.dumps(feats))
    return feats


def _geojson_subset(fn, geo, R):
    d = json.loads((CACHE / fn).read_text())
    lat0, lon0 = geo.lat, geo.lon
    dlat = (R + 60) / 111_000
    dlon = (R + 60) / (111_000 * math.cos(math.radians(lat0)))
    out = []
    for f in d["features"]:
        g = f.get("geometry")
        if not g:
            continue
        c = g["coordinates"]
        # quick bbox test on first coordinate
        flat = c
        try:
            while isinstance(flat[0], list):
                flat = flat[0]
            lon, lat = flat[0], flat[1]
        except (IndexError, TypeError):
            continue
        if abs(lat - lat0) < dlat and abs(lon - lon0) < dlon:
            out.append({"geometry": g, "properties": f.get("properties", {})})
    return out


def _to_plan(geom, geo):
    from shapely.ops import transform

    def fn(x, y, z=None):
        p = geo.lonlat_to_plan(np.asarray(x), np.asarray(y))
        return p[:, 0], p[:, 1]
    return transform(fn, geom)


def load(spec) -> SiteData:
    geo = Geo(spec)
    R = float(spec["site"]["extent"]["radius_m"])
    disk = Point(0, 0).buffer(R + 40)

    def feats(name):
        fs = _cached_subset(name, geo, R, lambda: _geojson_subset(FILES[name], geo, R))
        out = []
        for f in fs:
            try:
                g = _to_plan(shape(f["geometry"]), geo)
            except Exception:
                continue
            if not g.is_valid:
                g = g.buffer(0)
            if g.intersects(disk):
                out.append((g, f["properties"]))
        return out

    buildings = feats("buildings")
    sidewalks = feats("sidewalks")
    streets = feats("streets")
    parcels = feats("parcels")

    def trees_loader():
        rows = []
        lat0, lon0 = geo.lat, geo.lon
        with open(CACHE / FILES["street_trees"], encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    lon, lat = float(r["x_longitude"]), float(r["y_latitude"])
                except (ValueError, KeyError):
                    continue
                if abs(lat - lat0) < (R + 60) / 111_000 and abs(lon - lon0) < (R + 60) / 82_000:
                    rows.append({"lon": lon, "lat": lat, "spp": r.get("spp_com"), "dbh": r.get("dbh")})
        return rows
    tr = _cached_subset("street_trees", geo, R, trees_loader)
    street_trees = []
    for t in tr:
        p = geo.lonlat_to_plan(t["lon"], t["lat"])[0]
        street_trees.append((p[0], p[1], t))

    # DEM (UTM) -> plan grid, z relative to L1 FFE
    d = np.load(CACHE / "dem.npz")
    from rasterio.transform import Affine
    tr_ = Affine(*d["transform"])
    res = float(spec["site"]["extent"]["terrain_cell_m"])
    n = int(2 * (R + 30) / res)
    x0 = y0 = -(R + 30)
    gx, gy = np.meshgrid(x0 + (np.arange(n) + 0.5) * res, y0 + (np.arange(n) + 0.5) * res)
    E = geo.plan_to_crs("EPSG:26919", np.c_[gx.ravel(), gy.ravel()])
    col, row = ~tr_ * (E[:, 0], E[:, 1])
    z = d["z"]
    r_ = Raster(z, 0, 0, 1)  # raster in pixel space
    zz = Raster(np.asarray(z, float), -0.5, -0.5, 1.0).sample(col, row).reshape(n, n)
    dem = Raster(zz - geo.z0, x0, y0, res)

    o = np.load(CACHE / "ortho_filled.npz")
    # tiles z were stored relative to the Revit L1 (13.614 m); re-base
    rebase = spec["project"]["site"]["revit_export_level1_mm"] / 1000.0 - geo.z0
    top = o["top"] + rebase
    ortho = Ortho(o["img"], top, float(o["H"]), float(o["res"]))
    H, ores = float(o["H"]), float(o["res"])
    dsm = Raster(top, -H, -H, ores)
    return SiteData(geo, R, buildings, sidewalks, streets, parcels, street_trees, dem, ortho, dsm)
