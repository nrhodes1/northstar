"""Exterior gallery (shared porch), stairs, canopy, balconies, guardrails."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

from .. import geom as G
from ..model import Model
from ..units import FT, L, P, fmt_ftin
from .context import Ctx


# ------------------------------------------------------------- guardrails
def guardrail_path(model: Model, pts3, g: dict, level=None, system="gallery", base_follow=False):
    """Guardrail along a 3D polyline (base points). Posts, top & bottom rail,
    vertical pickets (or mesh panel)."""
    h = L(g["height"])
    mat = g["material"]
    tr = L(g["top_rail"]["size"][0])
    pk = L(g["picket"]["size"])
    sp = L(g["picket"]["spacing"]) + pk
    post_sp = L(g["post_spacing"])
    parts = []
    pts3 = [np.asarray(p, float) for p in pts3]
    for a, b in zip(pts3[:-1], pts3[1:]):
        v = b - a
        hl = np.linalg.norm(v[:2])
        if hl < 0.05:
            continue
        # rails
        parts.append(G.bar3d(a + [0, 0, h - tr / 2], b + [0, 0, h - tr / 2], tr, tr))
        parts.append(G.bar3d(a + [0, 0, 0.1], b + [0, 0, 0.1], tr * 0.6, tr * 0.6))
        # posts
        n_post = max(1, math.ceil(hl / post_sp))
        for k in range(n_post + 1):
            p = a + v * (k / n_post)
            parts.append(G.box3(p[0] - tr / 2, p[1] - tr / 2, p[2], p[0] + tr / 2, p[1] + tr / 2, p[2] + h))
        if g.get("type", "vertical_pickets") == "vertical_pickets":
            n = max(1, int(hl / sp))
            for k in range(1, n):
                p = a + v * (k / n)
                parts.append(G.box3(p[0] - pk / 2, p[1] - pk / 2, p[2] + 0.1, p[0] + pk / 2, p[1] + pk / 2,
                                    p[2] + h - tr))
        else:
            d = G.unit(v[:2])
            parts.append(G.oriented_box(a[:2], d, 0, hl, 0.005, -0.005, a[2] + 0.1, a[2] + h - tr))
    model.add(system, "guardrails", "IfcRailing", mat, G.concat(parts), level=level)


def rail_lines(region, exclude, min_len=0.3):
    """Edges of `region` that are not inside `exclude` (walls, stairs)."""
    lines = []
    for poly in G.as_polys(region):
        ln = LineString(poly.exterior.coords)
        rest = ln.difference(exclude) if exclude is not None else ln
        for s in getattr(rest, "geoms", [rest]):
            if s.length > min_len:
                lines.append(list(s.coords))
    return lines


# ------------------------------------------------------------- gallery
def build_gallery(ctx: Ctx, model: Model):
    sp = ctx.spec
    gl = sp["gallery"]
    rail = gl["guardrail"]
    deck = G.polygon([P(p) for p in gl["deck"]["outline"]])
    stairs = [(s, [np.array(P(p)) for p in s["footprint"]]) for s in gl["stairs"]]
    stair_polys = unary_union([G.polygon(q) for _, q in stairs])
    canopy_z = sp.level_z(gl["canopy_level"])
    thk = L(gl["deck"]["thickness"])

    # columns
    cs = L(gl["columns"]["size"])
    cols = []
    for p in gl["columns"]["points"]:
        x, y = P(p)
        cols.append(G.box3(x - cs / 2, y - cs / 2, -L("6\""), x + cs / 2, y + cs / 2, canopy_z - L("1\"")))
    model.add("gallery", "columns", "IfcColumn", gl["columns"]["material"], G.concat(cols), level="L1",
              count=len(cols))

    # decks
    for lv in gl["levels_served"]:
        z = sp.level_z(lv)
        surf = gl["deck"]["surface_material"]
        dt = L("1 1/2\"")
        deck_l = deck.difference(stair_polys.buffer(-0.01))
        model.add("gallery", "deck/surface", "IfcSlab", surf, G.extrude(deck_l, z - dt, z), level=lv)
        # framing: perimeter beams + joists perpendicular to the wall
        frame = []
        band = G.ring(deck, 0, L("4\""))
        frame.append(G.extrude(band, z - thk, z - dt))
        minx, miny, maxx, maxy = deck.bounds
        x = minx + L("2'-0\"")
        while x < maxx:
            j = LineString([(x, miny - 1), (x, maxy + 1)]).intersection(deck)
            for s in getattr(j, "geoms", [j]):
                if s.length > 0.3:
                    (xa, ya), (xb, yb) = s.coords[0], s.coords[-1]
                    frame.append(G.segment_solid((xa, ya), (xb, yb), L("2\""), z - thk + L("2\""), z - dt))
            x += L("2'-0\"")
        model.add("gallery", "deck/framing", "IfcBeam", gl["deck"]["framing_material"], G.concat(frame), level=lv)
        # rails: outer edges except against building and at stair heads
        bldg = ctx.outlines.get(lv)
        excl = unary_union([bldg.buffer(0.25), stair_polys.buffer(0.05)])
        for ln in rail_lines(deck, excl):
            guardrail_path(model, [(x, y, z) for x, y in ln], rail, level=lv)

    # stairs
    sd = gl["stair_detail"]
    for s, q in stairs:
        _stair(ctx, model, s, q, sd, rail)

    # canopy
    cp = gl["canopy"]
    cover = unary_union([deck, stair_polys]).buffer(L(cp["overhang"]), join_style=2)
    above = ctx.outlines.get(gl["canopy_level"])
    below = ctx.outlines.get(sp.level_above("L2") if False else "L3")
    cover = cover.difference(below.buffer(-0.01) if below is not None else cover.buffer(0))
    zc = canopy_z
    t = L(cp["thickness"])
    model.add("gallery", "canopy/frame", "IfcRoof", cp["frame_material"],
              G.extrude(G.ring(cover, 0, L("6\"")), zc - t, zc), level=gl["canopy_level"])
    # joists
    js = []
    minx, miny, maxx, maxy = cover.bounds
    x = minx + L("2'-9\"")
    while x < maxx:
        j = LineString([(x, miny - 1), (x, maxy + 1)]).intersection(cover)
        for s_ in getattr(j, "geoms", [j]):
            if s_.length > 0.3:
                js.append(G.segment_solid(s_.coords[0], s_.coords[-1], L("2\""), zc - t, zc - L("1\"")))
        x += L("3'-6\"")
    model.add("gallery", "canopy/frame", "IfcBeam", cp["frame_material"], G.concat(js), level=gl["canopy_level"])
    if cp["surface"] == "pv_canopy":
        w, ln = L(cp["pv_module"]["width"]), L(cp["pv_module"]["length"])
        gap = L(cp["pv_module"]["gap"])
        mods = []
        inner = G.inset(cover, L("3\""))
        y = miny
        while y < maxy:
            x = minx
            while x < maxx:
                from shapely.geometry import box as sbox
                r = sbox(x, y, x + w, y + ln)
                if inner.contains(r):
                    mods.append(G.extrude(r, zc, zc + L("1 3/8\"")))
                x += w + gap
            y += ln + gap
        model.add("gallery", "canopy/pv", "IfcSolarDevice", "pv_module", G.concat(mods), level=gl["canopy_level"],
                  count=len(mods))
        model.flag(f"Gallery canopy PV: {len(mods)} modules")
    else:
        model.add("gallery", "canopy/roof", "IfcRoof", cp["surface"],
                  G.extrude(cover, zc, zc + L("1\"")), level=gl["canopy_level"])


def _stair(ctx, model, s, q, sd, rail):
    sp = ctx.spec
    q0, q1, q2, q3 = q
    wdir = q1 - q0
    W = np.linalg.norm(wdir)
    wdir = wdir / W
    rdir = ((q3 - q0) + (q2 - q1)) / 2
    R = np.linalg.norm(rdir)
    rdir = rdir / R
    sw = min(L(sd["width"]), W / 2)
    landing = sw
    flight_len = R - landing
    levels = list(sp.levels)
    i0, i1 = levels.index(s["from_level"]), levels.index(s["to_level"])
    first_outer = s.get("first_flight_side", "outer") == "outer"
    tread_t = L("1 1/4\"")
    treads, stringers, landings = [], [], []
    for li in range(i0, i1):
        za, zb = sp.level_z(levels[li]), sp.level_z(levels[li + 1])
        H = zb - za
        n = math.ceil(H / L(sd["riser_max"]))
        if n % 2:
            n += 1
        rh = H / n
        per = n // 2
        tr = flight_len / (per - 1) if per > 1 else flight_len
        sides = [(W - sw, W), (0, sw)] if first_outer else [(0, sw), (W - sw, W)]
        zmid = za + rh * per
        # flight 1: from deck edge outward, rising
        for k, (s0, s1) in enumerate(sides):
            for t_ in range(per - (0 if k == 0 else 0)):
                if k == 0:
                    r0 = t_ * tr
                    zt = za + rh * (t_ + 1)
                else:
                    r0 = flight_len - (t_ + 1) * tr
                    zt = zmid + rh * (t_ + 1)
                if t_ == per - 1 and k == 0:
                    continue   # last riser lands on landing
                if t_ == per - 1 and k == 1:
                    continue   # last riser lands on deck
                corner = q0 + wdir * s0 + rdir * r0
                treads.append(G.oriented_box(corner, rdir, 0, tr + L("1\""), 0, -(s1 - s0), zt - tread_t, zt))
            # stringers both sides of the flight
            for off in (s0 + L("1\""), s1 - L("1\"")):
                if k == 0:
                    p0 = q0 + wdir * off
                    p1 = q0 + wdir * off + rdir * flight_len
                    z0_, z1_ = za, zmid
                else:
                    p0 = q0 + wdir * off + rdir * flight_len
                    p1 = q0 + wdir * off
                    z0_, z1_ = zmid, zb
                stringers.append(G.bar3d(np.r_[p0, z0_ - 0.12], np.r_[p1, z1_ - 0.12], L("2\""), L("12\"")))
            # rails on the outside of each flight
            off = s1 if (s0 == 0) is False else s0
            off = s0 if s0 == 0 else s1
            if k == 0:
                pa = np.r_[q0 + wdir * off, za]
                pb = np.r_[q0 + wdir * off + rdir * flight_len, zmid]
            else:
                pa = np.r_[q0 + wdir * off + rdir * flight_len, zmid]
                pb = np.r_[q0 + wdir * off, zb]
            guardrail_path(model, [pa, pb], rail, level=levels[li], system="gallery")
        # landing at far end, full width
        lc = q0 + rdir * flight_len
        landings.append(G.oriented_box(lc, rdir, 0, landing, 0, -W, zmid - L("6\""), zmid))
        # landing rail (far edge + sides)
        e0 = np.r_[q0 + rdir * R, zmid]
        e1 = np.r_[q1 + rdir * 0 + (q2 - q1) * 1.0, zmid]
        guardrail_path(model, [np.r_[q0 + rdir * flight_len, zmid], e0, np.r_[q0 + wdir * W + rdir * R, zmid],
                               np.r_[q0 + wdir * W + rdir * flight_len, zmid]], rail, level=levels[li])
    model.add("gallery", f"stairs/treads", "IfcStairFlight", sd["tread_material"], G.concat(treads), stair=s["name"])
    model.add("gallery", f"stairs/stringers", "IfcMember", sd["stringer"]["material"], G.concat(stringers),
              stair=s["name"])
    model.add("gallery", f"stairs/landings", "IfcSlab", sd["landing_material"], G.concat(landings), stair=s["name"])
    model.flag(f"Stair {s['name']}: width {fmt_ftin(sw)}, run {fmt_ftin(flight_len)} per flight")


# ------------------------------------------------------------- balconies
def build_balconies(ctx: Ctx, model: Model):
    sp = ctx.spec
    bs = sp["balconies"]
    dflt = bs["defaults"]
    rail = sp["gallery"]["guardrail"] if dflt["guardrail"] == "gallery" else dflt["guardrail"]
    for b in bs["balconies"]:
        st = b["level"]
        z = sp.level_z(st) + L(b.get("raised", 0))
        poly = G.polygon([P(p) for p in b["outline"]])
        # keep the deck outside the wall assembly
        wall_zone = G.inset(ctx.outlines[st], ctx.wall_T)
        deck = poly.difference(wall_zone)
        dt = L("1 1/2\"")
        model.add("balconies", "deck/surface", "IfcSlab", b.get("surface_material", dflt["surface_material"]),
                  G.extrude(deck, z - dt, z), level=st, balcony=b["id"])
        model.add("balconies", "deck/framing", "IfcBeam", dflt["framing_material"],
                  G.extrude(G.ring(deck, 0, L("3\"")), z - L(dflt["thickness"]), z - dt), level=st, balcony=b["id"])
        excl = ctx.outlines[st].buffer(0.12, join_style=2)
        for ln in rail_lines(deck, excl):
            guardrail_path(model, [(x, y, z) for x, y in ln], rail, level=st, system="balconies")
