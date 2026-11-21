"""Seed spec/openings.yaml from the Revit export embedded in the site .3dm.

One-time helper (re-runnable). The SD PDF shows openings in plan but not
sill/head heights; the Revit export in the client's site model carries the
same design with exact family sizes. Openings are snapped to the exterior
outline in spec/massing.yaml and written as plan points in feet.

    python tools/seed_openings_from_revit.py path/to/site.3dm
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import rhino3dm as r
import yaml
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from northstar import spec as specmod  # noqa: E402
from northstar.builders.context import Ctx  # noqa: E402
from northstar.units import FT  # noqa: E402

TYPE_BY_FAMILY = [
    ("Window - Single - Operable", "W1"),
    ("Window - Double - Operable", "W2"),
    ("Window-Fixed - 24", "W3"),
    ("ROOMS DOOR", "D1"),
    ("Door-Double-Sliding", "D2"),
    ("System Panel - Glazed", "SF"),
]


def main(path3dm, out="spec/openings.yaml"):
    sp = specmod.load()
    ctx = Ctx.build(sp)
    site = sp["project"]["site"]
    th = math.radians(site["plan_rotation_deg"])
    t = np.array(site["plan_origin_mm"])
    z0 = site["revit_export_level1_mm"]
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])

    def to_plan(x, y):
        return (R.T @ (np.array([x, y]) - t)) / 1000.0  # metres

    m = r.File3dm.Read(path3dm)
    idefs = {d.Id: d for d in m.InstanceDefinitions}
    levels = sp.levels
    lvl_names = [k for k in levels if k in ctx.outlines]
    found = {k: [] for k in lvl_names}
    for o in m.Objects:
        g = o.Geometry
        if not isinstance(g, r.InstanceReference):
            continue
        d = idefs.get(g.ParentIdefId)
        name = d.Name if d else ""
        typ = next((t_ for key, t_ in TYPE_BY_FAMILY if key in name), None)
        if not typ:
            continue
        bb = None
        for oid in d.GetObjectIds():
            oo = m.Objects.FindId(oid)
            if oo is None:
                continue
            b = oo.Geometry.GetBoundingBox()
            bb = b if bb is None else r.BoundingBox.Union(bb, b)
        pts = []
        for cx in (bb.Min.X, bb.Max.X):
            for cy in (bb.Min.Y, bb.Max.Y):
                for cz in (bb.Min.Z, bb.Max.Z):
                    p = r.Point3d(cx, cy, cz)
                    p.Transform(g.Xform)
                    pts.append((p.X, p.Y, p.Z))
        pts = np.array(pts)
        plan = np.array([to_plan(x, y) for x, y in pts[:, :2]])
        zmin = (pts[:, 2].min() - z0) / 1000.0
        zmax = (pts[:, 2].max() - z0) / 1000.0
        # storey by sill height
        st = None
        for k in lvl_names:
            if levels[k] - 0.3 <= zmin < levels[k] + 2.6:
                st = k
        if st is None:
            continue
        c = plan.mean(axis=0)
        # nearest exterior edge
        best = None
        for poly in [p for _, p in ctx.parts[st]]:
            ext = poly.exterior
            dist = ext.distance(Point(c))
            if best is None or dist < best[0]:
                best = (dist, ext)
        if best[0] > 0.75:      # not on an exterior wall (interior door etc.)
            continue
        ext = best[1]
        s = ext.project(Point(c))
        q = ext.interpolate(s)
        # extent along the wall direction
        a = np.array(ext.interpolate(max(s - 0.05, 0)).coords[0])
        b = np.array(ext.interpolate(s + 0.05).coords[0])
        dvec = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        proj = plan @ dvec
        width = proj.max() - proj.min()
        sill = zmin - levels[st]
        head = zmax - levels[st]
        found[st].append({
            "type": typ,
            "at": [float(round(q.x / FT, 3)), float(round(q.y / FT, 3))],
            "_w": width / FT, "_sill": sill / FT, "_head": head / FT, "_fam": name.split("-Working")[0][:60],
        })
    types = sp["glazing"]["types"]
    from northstar.units import L
    out_doc = {"openings": {}}
    for st, lst in found.items():
        rows = []
        seen = set()
        for o in sorted(lst, key=lambda o: (o["at"][1], o["at"][0])):
            key = (o["type"], round(o["at"][0], 1), round(o["at"][1], 1))
            if key in seen:
                continue
            seen.add(key)
            row = {"type": o["type"], "at": o["at"]}
            ty = types[o["type"]]
            if o["type"] == "SF":
                row["width"] = float(round(o["_w"], 3))
            if o["type"].startswith("W") and abs(o["_sill"] - L(ty["sill"]) / FT) > 0.1:
                dz = float(round(o["_sill"] - L(ty["sill"]) / FT, 3))
                row["raise"] = dz
            rows.append(row)
        out_doc["openings"][st] = rows
        print(st, len(rows))
    header = (
        "# ---------------------------------------------------------------------------\n"
        "# OPENINGS — every window/door in the exterior walls\n"
        "# ---------------------------------------------------------------------------\n"
        "# type -> glazing.yaml types. at = [x, y] ft on the exterior face (drawing\n"
        "# frame); the builder snaps to the nearest wall. Optional: width (ft),\n"
        "# raise (ft, shifts sill+head), sill/head overrides (ft).\n"
        "# Seeded by tools/seed_openings_from_revit.py — now hand-maintained.\n"
        "# ---------------------------------------------------------------------------\n"
    )
    with open(out, "w") as f:
        f.write(header)
        yaml.safe_dump(out_doc, f, sort_keys=False, default_flow_style=None, width=120)


if __name__ == "__main__":
    main(sys.argv[1])
