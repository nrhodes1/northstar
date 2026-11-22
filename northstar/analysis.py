"""Design-review analysis: computes located concerns for drawings + checks.

Each concern: {id, storey, kind, severity, xy (m, drawing frame), title, text}
severity: "code" (likely non-compliant), "coord" (needs coordination /
design decision), "info" (be aware).
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from . import geom as G
from .units import FT, L, P, fmt_ftin

SF = FT * FT


# ------------------------------------------------------------ fire separation
# IBC 2021 Table 705.9 (sprinklered, unprotected openings) - max % of wall area
FSD_TABLE = [(0.0, 0.00), (3.0, 0.15), (5.0, 0.25), (10.0, 0.45), (15.0, 0.75), (20.0, 1.01)]


def allowed_openings(fsd_ft):
    pct = 0.0
    for lim, p in FSD_TABLE:
        if fsd_ft >= lim:
            pct = p
    return pct


def fsd_lines(ctx):
    """Lines that fire separation distance is measured to: interior lot lines
    and the centerline of streets the lot fronts on (IBC 202)."""
    sp = ctx.spec
    lot = sp["site"].get("lot")
    if not lot:
        return [], []
    poly = G.polygon([P(p) for p in lot["points"]])
    centerlines = []
    for st in sp["site"].get("added_streets") or []:
        centerlines.append(LineString([P(p) for p in st["centerline"]]))
    try:   # city street centerlines, if the site cache is present
        from .site import data
        sd = data.load(sp)
        for ln, props in sd.streets:
            if ln.distance(poly) < 25:
                centerlines.append(ln)
    except Exception:
        pass
    interior, street = [], []
    for a, b in G.edges(poly):
        seg = LineString([a, b])
        mid = seg.interpolate(0.5, normalized=True)
        n = G.outward_normal(a, b)
        probe = Point(mid.x + n[0] * 3, mid.y + n[1] * 3)
        near = min((c.distance(probe) for c in centerlines), default=1e9)
        (street if near < 9.0 else interior).append(seg)
    return interior, centerlines if street else []


def fire_separation(ctx, openings):
    """Per storey and facade (edges grouped by facing direction, IBC 705.8):
    allowed opening area = sum over wall segments of (segment area x Table
    705.9 % at that segment's FSD); compare with the actual opening area."""
    interior, centerlines = fsd_lines(ctx)
    targets = interior + centerlines
    out = []
    if not targets:
        return out
    by_edge = {}
    for o in openings:
        key = (o["storey"], tuple(np.round(o["a"], 3)), tuple(np.round(o["b"], 3)))
        by_edge.setdefault(key, []).append(o)
    for st, parts in ctx.parts.items():
        z0, z1, _ = ctx.storey_span(st)
        faces = {}
        for _, poly in parts:
            for a, b in G.edges(poly):
                Lg = float(np.linalg.norm(b - a))
                if Lg < 0.3:
                    continue
                n = G.outward_normal(a, b)
                best = 1e9
                for t in (0.1, 0.5, 0.9):
                    p = a + (b - a) * t
                    ray = LineString([p, p + n * 60])
                    for tg in targets:
                        x = ray.intersection(tg)
                        if not x.is_empty:
                            best = min(best, Point(p).distance(x))
                fsd_ft = best / FT
                if fsd_ft > 20:
                    continue
                key = (st, tuple(np.round(a, 3)), tuple(np.round(b, 3)))
                op = sum(o["w"] * (o["head"] - o["sill"]) for o in by_edge.get(key, []))
                wall = Lg * (z1 - z0)
                face = int(round(math.degrees(math.atan2(n[1], n[0])) / 15.0)) * 15 % 360
                f = faces.setdefault(face, {"wall": 0, "open": 0, "allow": 0, "fsd": 1e9, "pts": []})
                f["wall"] += wall
                f["open"] += op
                f["allow"] += wall * allowed_openings(fsd_ft)
                f["fsd"] = min(f["fsd"], fsd_ft)
                f["pts"].append(((a + b) / 2 + n * 0.8, Lg))
        for face, f in faces.items():
            pct, allow = f["open"] / f["wall"], f["allow"] / f["wall"]
            xy = max(f["pts"], key=lambda t: t[1])[0]
            sev = "code" if f["open"] > f["allow"] + 1e-6 else "info"
            out.append(dict(
                storey=st, kind="fire_separation", severity=sev, xy=tuple(xy),
                title=f"FSD {f['fsd']:.1f}' - openings {pct:.0%} of face (allowed {allow:.0%})",
                text=(f"Facade is {f['fsd']:.1f}'+ from an interior lot line. IBC Table 705.9 (sprinklered, "
                      f"unprotected) allows {allow:.0%} openings on this face at this storey; it has {pct:.0%}. "
                      + ("Reduce/protect openings (rated glazing or opening protectives) or adjust the lot line. "
                         if sev == "code" else "")
                      + "Wall needs a fire-resistance rating per Table 705.5 where FSD < 10'.")))
    return out


# ------------------------------------------------------------ plumbing stacks
def wet_stacking(ctx, tol_ft=4.0):
    fx = (ctx.spec.get("fixtures") or {}).get("fixtures") or {}
    names = list(ctx.outlines)
    out = []
    for i, st in enumerate(names[1:], 1):
        below = fx.get(names[i - 1], [])
        for f in fx.get(st, []):
            if f["type"] not in ("WC", "tub", "shower"):
                continue
            p = np.array(P(f["at"]))
            dists = [np.linalg.norm(p - np.array(P(g["at"]))) for g in below if g["type"] in ("WC", "tub", "shower")]
            d = min(dists) / FT if dists else 99
            if d > tol_ft:
                out.append(dict(storey=st, kind="wet_stack", severity="coord", xy=tuple(p),
                                title=f"{f['type']} not stacked ({d:.0f}' offset)",
                                text=f"No wet fixture within {tol_ft:.0f}' on {names[i - 1]} below. Needs a "
                                     f"horizontal drain run in the floor zone (CLT penetrations, slope, acoustic) "
                                     f"- see A3.00 redline on floor-to-floor."))
    return out


# ------------------------------------------------------------ envelope
def odd_corners(ctx):
    out = []
    for st, parts in ctx.parts.items():
        for _, poly in parts:
            c = np.asarray(poly.exterior.coords)[:-1]
            for i in range(len(c)):
                a, b, d = c[i - 1], c[i], c[(i + 1) % len(c)]
                u, v = G.unit(a - b), G.unit(d - b)
                ang = math.degrees(math.acos(np.clip(np.dot(u, v), -1, 1)))
                if abs(ang - 90) > 3 and abs(ang - 180) > 3:
                    out.append(dict(storey=st, kind="corner", severity="coord", xy=tuple(b),
                                    title=f"{ang:.0f} deg corner",
                                    text="Non-orthogonal straw-panel corner - confirm New Frameworks corner "
                                         "detail/mitre (A1.01 redline)."))
    return out


def openings_near_corners(ctx, openings):
    mn = L(ctx.spec["envelope"]["panelization"]["min_width"])
    out = []
    for o in openings:
        Lg = float(np.linalg.norm(o["b"] - o["a"]))
        left = o["u"] - o["w"] / 2
        right = Lg - (o["u"] + o["w"] / 2)
        edge = min(left, right)
        if edge < mn + ctx.wall_T:   # panel pier between corner & opening is too narrow
            d = G.unit(o["b"] - o["a"])
            xy = o["a"] + d * o["u"]
            out.append(dict(storey=o["storey"], kind="pier", severity="coord", xy=tuple(xy),
                            title=f"{o['type']} {fmt_ftin(max(edge, 0))} from corner",
                            text=f"Straw-panel pier between corner and opening < {fmt_ftin(mn)} + wall depth: "
                                 f"site-built infill or shift opening."))
    return out


def area_loss(ctx, drawn_wall="16\""):
    """Net interior area lost to the thicker assembly vs the drawn wall."""
    rows = []
    for st, g in ctx.outlines.items():
        a0 = G.inset(g, L(drawn_wall)).area
        a1 = G.inset(g, ctx.wall_T).area
        rows.append((st, (a0 - a1) / SF))
    return rows


# ------------------------------------------------------------ structure/stairs
def structure_flags(model):
    out = []
    for e in model.elements:
        if e.layer == "posts" and not e.props.get("in_wall"):
            c = e.mesh.centroid
            out.append(dict(storey=e.level, kind="post", severity="coord", xy=(c[0], c[1]),
                            title="Post in room",
                            text="Glulam post on the bearing line can't be hidden in a stacked partition - "
                                 "move partition, add transfer, or deepen beam."))
    return out


def stair_flags(ctx):
    sp = ctx.spec
    sd = sp["gallery"]["stair_detail"]
    code = sp["code"]["stairs"]
    out = []
    for s in sp["gallery"]["stairs"]:
        q = [np.array(P(p)) for p in s["footprint"]]
        W = np.linalg.norm(q[1] - q[0])
        R = np.linalg.norm(((q[3] - q[0]) + (q[2] - q[1])) / 2)
        sw = min(L(sd["width"]), W / 2)
        H = L(10)
        n = math.ceil(H / L(sd["riser_max"]))
        n += n % 2
        tread = (R - sw) / (n // 2 - 1)
        if tread < L(code["min_tread"]) - 1e-3:
            need = (L(code["min_tread"]) * (n // 2 - 1) + sw) - R
            for st in ("L1", "L2", "L3"):
                out.append(dict(storey=st, kind="stair", severity="code", xy=tuple(np.mean(q, axis=0)),
                                title=f"Stair tread {fmt_ftin(tread)} < 11\"",
                                text=f"{s['name']}: footprint {fmt_ftin(R)} deep fits 2 x {n // 2} risers + "
                                     f"{fmt_ftin(sw)} landing only with short treads. Lengthen by "
                                     f"{fmt_ftin(need)} or reduce landing (IBC 1011.5.2)."))
    return out


def all_concerns(model, ctx, openings):
    cs = (fire_separation(ctx, openings) + stair_flags(ctx) + structure_flags(model) + wet_stacking(ctx)
          + odd_corners(ctx) + openings_near_corners(ctx, openings))
    # de-duplicate fire-separation 'info' noise: keep code items + one info per storey/face
    for i, c in enumerate(cs):
        c["id"] = i + 1
    return cs
