"""Seed spec/rooms.yaml from the room tags on A1.01-A1.04 (name, number,
area) and spec/fixtures.yaml from the plumbing fixtures in the Revit export.

    python tools/extract_rooms.py sources/<SD>.pdf [path/to/site.3dm]
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pymupdf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.extract_pdf import GRIDA_Y, GRID1_X, K, PAGES  # noqa: E402


def rooms(pdf):
    doc = pymupdf.open(pdf)
    out = {}
    for st, pg in PAGES.items():
        ws = doc[pg].get_text("words")
        rows = []
        for w in ws:
            if not re.fullmatch(r"\d{3}[A-C]?", w[4]):
                continue
            cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
            name = [v for v in ws if abs((v[0] + v[2]) / 2 - cx) < 40 and -22 < v[3] - cy < -4 and v[4] != "SF"]
            area = [v for v in ws if abs((v[0] + v[2]) / 2 - cx) < 40 and 4 < v[1] - cy < 22]
            nm = " ".join(v[4] for v in sorted(name, key=lambda v: (round(v[1]), v[0])))
            sf = None
            for v in area:
                try:
                    sf = float(v[4].replace(",", ""))
                    break
                except ValueError:
                    pass
            rows.append({"id": w[4], "name": nm or "?", "area_sf": sf,
                         "at": [round((cx - GRID1_X) / K, 2), round(-(cy - GRIDA_Y) / K, 2)]})
        out[st] = sorted(rows, key=lambda r: r["id"])
    return out


def fixtures(path3dm):
    import numpy as np
    import rhino3dm as r
    from northstar import spec as specmod
    sp = specmod.load()
    s = sp["project"]["site"]
    th = math.radians(s["plan_rotation_deg"])
    t = np.array(s["plan_origin_mm"])
    z0 = s["revit_export_level1_mm"]
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    m = r.File3dm.Read(path3dm)
    idefs = {d.Id: d for d in m.InstanceDefinitions}
    kinds = {"Toilet": "WC", "Tub": "tub", "Sink Vanity": "lav", "Shower": "shower"}
    levels = sp.levels
    out = {}
    for o in m.Objects:
        g = o.Geometry
        if not isinstance(g, r.InstanceReference):
            continue
        d = idefs.get(g.ParentIdefId)
        k = next((v for key, v in kinds.items() if d and key in d.Name), None)
        if not k:
            continue
        p = r.Point3d(0, 0, 0)
        p.Transform(g.Xform)
        xy = (R.T @ (np.array([p.X, p.Y]) - t)) / 304.8
        z = (p.Z - z0) / 304.8 * 0.3048
        st = max((n for n in levels if levels[n] <= z + 0.3), key=lambda n: levels[n])
        out.setdefault(st, []).append({"type": k, "at": [round(float(xy[0]), 2), round(float(xy[1]), 2)]})
    return out


if __name__ == "__main__":
    rm = rooms(sys.argv[1])
    with open("spec/rooms.yaml", "w") as f:
        f.write("# Room tags from the SD plans (id, name, scheduled area, label point in ft).\n")
        yaml.safe_dump({"rooms": rm}, f, sort_keys=False, default_flow_style=None, width=140)
    print({k: len(v) for k, v in rm.items()})
    if len(sys.argv) > 2:
        fx = fixtures(sys.argv[2])
        with open("spec/fixtures.yaml", "w") as f:
            f.write("# Plumbing fixtures (from the Revit export) - used for wet-stack checks.\n")
            yaml.safe_dump({"fixtures": fx}, f, sort_keys=False, default_flow_style=None, width=140)
        print({k: len(v) for k, v in fx.items()})
