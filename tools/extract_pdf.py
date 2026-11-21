"""Extract grids + wall linework from the SD vector PDF.

Seeds spec/interiors/<storey>.yaml (interior partition polygons, ft) and
prints exterior-outline candidates used to author spec/massing.yaml.

    python tools/extract_pdf.py sources/261109_HFH_North_Star_House_SD_Progress.pdf

Method: heavy black strokes (1.59 pt = Revit cut lines) are buffered and
unioned; enclosed thin regions (2*A/P < 1.9 ft) are wall poché. Scale is
calibrated from grid 1-9 (8 x 11'-6").
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import yaml
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from northstar import spec as specmod  # noqa: E402
from northstar.builders.context import Ctx  # noqa: E402
from northstar.units import FT  # noqa: E402

PAGES = {"L1": 4, "L2": 5, "L3": 6, "L4": 7}
GRID1_X, GRIDA_Y = 554.31, 424.42           # sheet points (A1.01-A1.04)
K = (1797.54 - 554.31) / (8 * 11.5)         # points per foot (~13.51 = 3/16")


def walls_on_page(page):
    segs = []
    for x in page.get_drawings():
        if x["type"] != "s" or not x.get("color") or max(x["color"]) > 0.01:
            continue
        if abs((x.get("width") or 0) - 1.59) > 0.05:
            continue
        for it in x["items"]:
            if it[0] == "l":
                segs.append(LineString([(it[1].x, it[1].y), (it[2].x, it[2].y)]))
            elif it[0] == "re":
                r = it[1]
                segs.append(LineString([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)]))
            elif it[0] == "c":
                segs.append(LineString([(it[1].x, it[1].y), (it[4].x, it[4].y)]))
    u = unary_union([s.buffer(0.9, cap_style=2) for s in segs])
    holes = []
    for g in (u.geoms if isinstance(u, MultiPolygon) else [u]):
        for h in g.interiors:
            p = Polygon(h)
            if 2 * p.area / p.length / K < 1.9 and p.area > 4:
                holes.append(p.buffer(0.9, join_style=2))
    return unary_union(holes)


def to_ft(g):
    from shapely import affinity
    g = affinity.translate(g, -GRID1_X, -GRIDA_Y)
    return affinity.scale(g, 1 / K, -1 / K, origin=(0, 0))


def main(pdf):
    doc = pymupdf.open(pdf)
    sp = specmod.load()
    ctx = Ctx.build(sp)
    out_dir = Path("spec/interiors")
    out_dir.mkdir(exist_ok=True)
    for st, pg in PAGES.items():
        w = to_ft(walls_on_page(doc[pg]))
        outline_ft = to_ft_m(ctx.outlines[st])
        ext_band = outline_ft.difference(outline_ft.buffer(-1.75, join_style=2)).buffer(0.15)
        inner = w.difference(ext_band).intersection(outline_ft.buffer(-0.5))
        polys = []
        cols = []
        for p in (inner.geoms if hasattr(inner, "geoms") else [inner]):
            if not isinstance(p, Polygon) or p.area < 0.15:
                continue
            p = p.simplify(0.03)
            b = p.bounds
            if b[2] - b[0] < 1.0 and b[3] - b[1] < 1.0:
                cols.append([round(p.centroid.x, 3), round(p.centroid.y, 3)])
                continue
            polys.append([[round(x, 3), round(y, 3)] for x, y in list(p.exterior.coords)[:-1]])
        doc_out = {
            "storey": st,
            "source": f"A1.0{pg - 3} vector linework (tools/extract_pdf.py)",
            "height": "auto",   # floor-to-underside of structure above
            "material": "interior_partition",
            "walls": [{"outline": p} for p in polys],
        }
        with open(out_dir / f"{st}.yaml", "w") as f:
            f.write("# Interior partitions (plan polygons, ft). Seeded from the SD PDF.\n")
            yaml.safe_dump(doc_out, f, sort_keys=False, default_flow_style=None, width=140)
        print(st, len(polys), "partitions;", len(cols), "small solids skipped")


def to_ft_m(g):
    from shapely import affinity
    return affinity.scale(g, 1 / FT, 1 / FT, origin=(0, 0))


if __name__ == "__main__":
    main(sys.argv[1])
