"""Build the site: terrain, ground surfaces, context buildings, trees.

    python -m northstar.site.build      -> out/site.glb, out/site.3dm, out/site_scatter.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Point
from shapely.ops import unary_union

from .. import spec as specmod
from ..builders.context import Ctx
from ..model import Model
from . import context as ctxb
from . import data, terrain

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"


def our_lot(sd, ctx):
    """Property line from spec/site.yaml (A1.00); falls back to Boston parcels."""
    from ..geom import polygon
    from ..units import P
    lot = ctx.spec["site"].get("lot")
    bl = ctx.outlines["L1"]
    if lot:
        return polygon([P(p) for p in lot["points"]]).union(bl.buffer(0.3))
    lots = [p for p, _ in sd.parcels if p.intersects(bl) and p.intersection(bl).area > 1.0]
    lot = unary_union(lots).buffer(0) if lots else bl.buffer(3)
    return lot.union(bl.buffer(0.3))


def trees(sd, ctx, footprints, lot, grade, scatter):
    sp = ctx.spec
    tc = sp["context"]["trees"]
    have = []
    disk = Point(0, 0).buffer(sd.R)
    for x, y, t in sd.street_trees:
        if not disk.contains(Point(x, y)) or lot.contains(Point(x, y)):
            continue
        try:
            dbh = float(t.get("dbh") or 8)
        except ValueError:
            dbh = 8
        dsm_h = float(sd.dsm.sample(x, y) - sd.dem.sample(x, y))
        h = dsm_h if 3 < dsm_h < 30 else min(22, 3 + 0.42 * dbh)
        scatter["trees"].append({"x": x, "y": y, "z": float(grade.sample(x, y)), "h": h,
                                 "r": max(1.2, tc["crown_ratio"] * h), "species": t.get("spp") or "", "src": "bprd"})
        have.append((x, y))
    # canopy peaks from the tiles surface model
    from scipy.ndimage import maximum_filter
    H, res = sd.ortho.H, sd.ortho.res
    X, Y = np.meshgrid(-H + (np.arange(sd.ortho.top.shape[1]) + 0.5) * res,
                       -H + (np.arange(sd.ortho.top.shape[0]) + 0.5) * res)
    nd = sd.ortho.top - sd.dem.sample(X.ravel(), Y.ravel()).reshape(X.shape)
    veg = terrain.vegetation_mask(sd, sp)
    canopy = veg & (nd > tc["canopy_min_height_m"])
    win = int(tc["canopy_peak_spacing_m"] / res)
    peaks = canopy & (nd == maximum_filter(np.where(canopy, nd, -1), size=win))
    iy, ix = np.nonzero(peaks)
    fp = footprints.buffer(0.5)
    from shapely import prepare
    prepare(fp)
    pts = shapely.points(X[iy, ix], Y[iy, ix])
    ok = ~shapely.contains(fp, pts) & shapely.contains(disk, pts) & ~shapely.contains(lot, pts)
    for i in np.nonzero(ok)[0]:
        x, y, h = float(X[iy[i], ix[i]]), float(Y[iy[i], ix[i]]), float(nd[iy[i], ix[i]])
        if h > 30 or any((x - a) ** 2 + (y - b) ** 2 < 9 for a, b in have):
            continue
        scatter["trees"].append({"x": x, "y": y, "z": float(grade.sample(x, y)), "h": h,
                                 "r": max(1.5, tc["crown_ratio"] * h), "species": "", "src": "canopy"})
        have.append((x, y))


def build():
    t0 = time.time()
    sp = specmod.load()
    ctx = Ctx.build(sp)
    sd = data.load(sp)
    print(f"  data      {time.time() - t0:5.1f}s  ({len(sd.buildings)} bldgs, {len(sd.sidewalks)} sidewalks, "
          f"{len(sd.parcels)} parcels, {len(sd.street_trees)} street trees)")
    model = Model("site")
    lot = our_lot(sd, ctx)
    model.flag(f"Our lot: {lot.area / 0.0929:,.0f} SF (property line from A1.00)")
    scatter = {"trees": [], "shrubs": [], "boulders": []}
    t = time.time()
    # grade first (needed by context bases)
    grade = terrain.graded(sd, ctx, lot)
    footprints = ctxb.build_context(sd, ctx, model, lot, grade)
    print(f"  context   {time.time() - t:5.1f}s")
    t = time.time()
    grade, road, sidewalk = terrain.build_ground(sd, ctx, model, lot, footprints)
    terrain.build_markings(sd, ctx, model, grade, road, sidewalk)
    terrain.build_lot_landscape(sd, ctx, model, grade, lot, scatter)
    print(f"  ground    {time.time() - t:5.1f}s")
    t = time.time()
    trees(sd, ctx, footprints, lot, grade, scatter)
    print(f"  trees     {time.time() - t:5.1f}s  ({len(scatter['trees'])})")
    # lot boundary for Rhino
    model.curve("site::property_line", [(x, y, 0) for x, y in lot.exterior.coords])
    return model, ctx, scatter


def main():
    OUT.mkdir(exist_ok=True)
    model, ctx, scatter = build()
    from ..export import gltf, rhino
    mats = ctx.spec["materials"]
    gltf.write(model, OUT / "site.glb", materials=mats)
    rhino.write(model, ctx.spec, OUT / "northstar_site.3dm", frame="site", materials=mats)
    with open(OUT / "site_scatter.json", "w") as f:
        json.dump(scatter, f)
    with open(OUT / "site_report.md", "w") as f:
        f.write("# Site build\n\n" + "\n".join(f"- {n}" for n in model.notes) + "\n")
    print("\n".join(model.notes))
    print(f"{len(model.elements)} site elements -> {OUT}")


if __name__ == "__main__":
    main()
