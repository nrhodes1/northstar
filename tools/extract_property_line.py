"""Extract the property line from A1.00 and register it to the drawing frame.

The site plan (1/16" = 1'-0", true north up) draws the lot as a dashed 1.8 pt
line and the L1 walls as 0.65 pt lines. The walls are fitted to the L1
outline in spec/massing.yaml (rotation + translation, scale fixed), and the
same transform is applied to the property line. Writes `lot` into
spec/site.yaml.

    python tools/extract_property_line.py sources/261109_HFH_North_Star_House_SD_Progress.pdf
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pymupdf
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from northstar import spec as specmod  # noqa: E402
from northstar.builders.context import Ctx  # noqa: E402
from northstar.units import FT  # noqa: E402

PAGE = 3
PT_PER_FT = 72 / 16   # 1/16" = 1'-0"


def lines(page, width):
    out = []
    for x in page.get_drawings():
        if x["type"] == "s" and x.get("color") and max(x["color"]) < 0.01 and abs((x.get("width") or 0) - width) < .02:
            for it in x["items"]:
                if it[0] == "l":
                    out.append(((it[1].x, -it[1].y), (it[2].x, -it[2].y)))   # y up
    return out


def main(pdf):
    page = pymupdf.open(pdf)[PAGE]
    # property line: chain of dashed segments -> polygon from their endpoints hull order
    pl = lines(page, 1.8)
    pts = np.array([p for s in pl for p in s])
    # order by angle around centroid (lot is star-shaped wrt its centroid)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    lot = Polygon(pts[np.argsort(ang)]).buffer(0).simplify(0.5)
    # walls: chamfer-fit the 0.65 pt wall lines to the L1 outline boundary
    sp = specmod.load()
    ctx = Ctx.build(sp)
    target = affinity.scale(ctx.outlines["L1"], 1 / FT, 1 / FT, origin=(0, 0))   # feet
    bnd = target.boundary
    W = np.array([p for s_ in lines(page, 0.65) for p in s_]) / PT_PER_FT
    from shapely import points as spoints, distance
    rot0 = -sp["project"]["site"]["plan_rotation_deg"]

    def cost(r, tx, ty):
        th = math.radians(r)
        c_, s_ = math.cos(th), math.sin(th)
        X = W[:, 0] * c_ - W[:, 1] * s_ + tx
        Y = W[:, 0] * s_ + W[:, 1] * c_ + ty
        d = distance(bnd, spoints(X, Y))
        return float(np.mean(np.minimum(d, 3.0)))

    th = math.radians(rot0)
    Wc = W.mean(axis=0)
    cx = Wc[0] * math.cos(th) - Wc[1] * math.sin(th)
    cy = Wc[0] * math.sin(th) + Wc[1] * math.cos(th)
    best = None
    for r in np.linspace(rot0 - 2, rot0 + 2, 17):
        for ex in np.arange(-12, 12.1, 1.0):
            for ey in np.arange(-12, 12.1, 1.0):
                tx, ty = target.centroid.x - cx + ex, target.centroid.y - cy + ey
                c = cost(r, tx, ty)
                if best is None or c < best[0]:
                    best = (c, r, tx, ty)
    # refine
    c0, r0, tx0, ty0 = best
    for r in np.linspace(r0 - 0.25, r0 + 0.25, 11):
        for ex in np.linspace(-1, 1, 11):
            for ey in np.linspace(-1, 1, 11):
                c = cost(r, tx0 + ex, ty0 + ey)
                if c < best[0]:
                    best = (c, r, tx0 + ex, ty0 + ey)
    iou, r, tx, ty = best
    dr = r - rot0
    L_ = affinity.scale(lot, 1 / PT_PER_FT, 1 / PT_PER_FT, origin=(0, 0))
    L_ = affinity.rotate(L_, rot0 + dr, origin=(0, 0), use_radians=False)
    L_ = affinity.translate(L_, tx, ty)
    coords = [[round(x, 2), round(y, 2)] for x, y in list(L_.exterior.coords)[:-1]]
    print(f"fit mean wall-to-outline distance {iou:.2f} ft, rotation {rot0 + dr:.2f} deg; lot {L_.area:,.0f} SF, {len(coords)} vertices")
    spath = Path("spec/site.yaml")
    s = spath.read_text()
    block = ("lot:\n  # Property line from A1.00 (registered to the L1 walls; mean residual %.2f ft).\n"
             "  # The SD redline notes the survey alignment is uncertain (D-018).\n  points: %s\n\n"
             % (iou, coords))
    s = re.sub(r"lot:\n(  .*\n)+\n", "", s)
    s = s.replace("grading:", block + "grading:", 1)
    spath.write_text(s)


if __name__ == "__main__":
    main(sys.argv[1])
