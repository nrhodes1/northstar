"""Floors (CLT + build-up), ground slab, roofs, terraces, soffits, PV."""
from __future__ import annotations

import math

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, box as sbox
from shapely.ops import unary_union

from .. import geom as G
from ..model import Model
from ..units import L, fmt_ftin
from .context import Ctx


def build_floors(ctx: Ctx, model: Model):
    sp = ctx.spec
    st_spec = sp["structure"]
    storeys = list(ctx.outlines)
    face = ctx.structure_face          # slabs bear to outer face of straw
    # ground slab
    first = storeys[0]
    gs = st_spec["floors"]["ground_slab"]
    region = G.inset(ctx.outlines[first], face)
    z = sp.level_z(first)
    fin = sum(L(x["thickness"]) for x in st_spec["floors"]["assembly"][:1])
    model.add("structure", "slabs/ground", "IfcSlab", gs["material"],
              G.extrude(region, z - fin - L(gs["thickness"]), z - fin), level=first)
    model.add("structure", "floor_finish", "IfcCovering", "wood_floor",
              G.extrude(G.inset(ctx.outlines[first], ctx.wall_T), z - fin, z), level=first)

    # suspended floors
    for st in storeys[1:]:
        below = ctx.storey_below(st)
        z = sp.level_z(st)
        region = G.inset(unary_union([ctx.outlines[st], ctx.outlines[below]]), face)
        dz = z
        for lay in st_spec["floors"]["assembly"]:
            t = L(lay["thickness"])
            if lay.get("structural"):
                reg = region
            else:  # finishes only inside the storey's own walls
                reg = G.inset(ctx.outlines[st], ctx.wall_T)
            if lay.get("material"):
                model.add("structure" if lay.get("structural") else "interiors",
                          "slabs/clt" if lay.get("structural") else f"floor_build_up/{lay['name']}",
                          "IfcSlab" if lay.get("structural") else "IfcCovering",
                          lay["material"], G.extrude(reg, dz - t, dz), level=st)
            dz -= t
        # soffit where this storey oversails the one below
        over = G.inset(unary_union([ctx.outlines[st], ctx.outlines[below]]), 0).difference(
            ctx.outlines[below].buffer(-0.01))
        over = over.buffer(-0.005)
        if over.area > 0.05:
            model.add("envelope", "soffits", "IfcCovering", sp["roofs"]["soffits"]["material"],
                      G.extrude(over, z - ctx.floor_T - L("3/4\""), z - ctx.floor_T), level=st)
            # the slab edge band in the cladding zone of the overhang
            band = G.ring(ctx.outlines[st], 0, face).intersection(over.buffer(0.02))
            model.add("envelope", "walls/cladding", "IfcCovering",
                      ctx.layers[0][3], G.extrude(band, z - ctx.floor_T - 0.02, z), level=st)

    # roof slab (top storey)
    top = storeys[-1]
    zr = sp.level_z(sp.level_above(top))
    region = G.inset(ctx.outlines[top], face)
    model.add("structure", "slabs/roof_clt", "IfcSlab", "clt",
              G.extrude(region, zr - ctx.roof_clt, zr), level="ROOF")


def build_roofs(ctx: Ctx, model: Model):
    sp = ctx.spec
    rs = sp["roofs"]
    storeys = list(ctx.outlines)
    top = storeys[-1]
    zr = sp.level_z(sp.level_above(top))
    # --- main roof build-up
    zone = G.inset(ctx.outlines[top], ctx.wall_T)
    dz = zr
    for lay in reversed([x for x in sp["structure"]["roof"]["assembly"] if not x.get("structural")]):
        t = L(lay["thickness"])
        if lay.get("material"):
            model.add("roof", f"main/{lay['name']}", "IfcCovering", lay["material"],
                      G.extrude(zone, dz, dz + t), level="ROOF")
        dz += t
    roof_top = dz
    if rs["main_roof"]["pv"]["enabled"]:
        for name, poly in ctx.parts[top]:
            usable = G.inset(poly, ctx.wall_T + L(rs["main_roof"]["setback_from_parapet"]))
            _pv_field(ctx, model, usable, roof_top, rs["main_roof"]["pv"], name)

    # --- terraces = exposed top of lower storeys
    ter = rs["terrace"]
    for st in storeys[:-1]:
        above = ctx.storey_above(st)
        z = sp.level_z(above)
        exposed = ctx.outlines[st].difference(ctx.outlines[above].buffer(0.02, join_style=2))
        exposed = G.inset(exposed, ctx.wall_T)
        exposed = exposed.buffer(-0.01).buffer(0.01)
        for poly in G.as_polys(exposed):
            if poly.area < 1.0:
                continue
            model.add("roof", "terrace/membrane", "IfcCovering", "membrane",
                      G.extrude(poly, z, z + L("1/4\"")), level=above)
            pav = _pavers(poly, L(ter["paver_size"][0]), L(ter["paver_size"][1]))
            model.add("roof", "terrace/pavers", "IfcCovering", ter["paver_material"],
                      G.extrude(pav, z + L(ter["buildup"]) - L("2\""), z + L(ter["buildup"])), level=above)
            if ter["planters"]["enabled"]:
                _planters(model, poly, z + L(ter["buildup"]), ter["planters"], above)
            model.flag(f"Terrace on {st} roof: {poly.area / 0.092903:.0f} SF")


def _pavers(poly, a, b, gap=0.006):
    minx, miny, maxx, maxy = poly.bounds
    tiles = []
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            tiles.append(sbox(x + gap / 2, y + gap / 2, x + a - gap / 2, y + b - gap / 2))
            x += a
        y += b
    return unary_union(tiles).intersection(poly.buffer(-0.01))


def _planters(model, poly, z, pl, level):
    """Linear planter along the longest edge of a terrace."""
    best = None
    for a, b in G.edges(G.polygon(poly.exterior.coords)):
        Lg = np.linalg.norm(b - a)
        if best is None or Lg > best[0]:
            best = (Lg, a, b)
    Lg, a, b = best
    if Lg < 2.0:
        return
    d = G.unit(b - a)
    dep = L(pl["depth"])
    box = G.oriented_box(a, d, 0.3, Lg - 0.3, 0.0, -dep, z, z + L(pl["height"]))
    inner = G.oriented_box(a, d, 0.35, Lg - 0.35, -0.05, -dep + 0.05, z + L(pl["height"]) - 0.15,
                           z + L(pl["height"]) + 0.1)
    model.add("roof", "terrace/planters", "IfcFurnishingElement", pl["material"], G.difference(box, [inner]),
              level=level)
    soil = G.oriented_box(a, d, 0.35, Lg - 0.35, -0.05, -dep + 0.05, z + L(pl["height"]) - 0.15,
                          z + L(pl["height"]) - 0.08)
    model.add("landscape", "planting/terrace", "IfcGeographicElement", pl["planting"], soil, level=level,
              scatter="meadow")


def _pv_field(ctx, model, usable, z_top, pv, name):
    """Herringbone PV layout clipped to `usable`."""
    w, ln = L(pv["module"]["width"]), L(pv["module"]["length"])
    t = L(pv["module"]["thickness"])
    gap = L(pv["gap"])
    rot = pv["rotation_deg"]
    tilt = math.radians(pv["tilt_deg"])
    rack = L(pv["rack_height"])
    if usable.is_empty:
        return
    c = usable.centroid
    # herringbone in a rotated frame: staircase strips of H (L x W) and
    # V (W x L) modules stepping (W, W); strips repeat every (L, -L).
    U = affinity.rotate(usable, -rot, origin=c)
    minx, miny, maxx, maxy = U.bounds
    Wg, Lg = w + gap, ln + gap
    span = (maxx - minx) + (maxy - miny) + 2 * Lg
    inner = U.buffer(-0.02)

    def layout(px, py):
        out = []
        for s_ in range(-int(span / Lg) - 2, int(span / Lg) + 3):
            ox, oy = minx + px + s_ * Lg, miny + py - s_ * Lg
            for i in range(-int(span / Wg) - 2, int(span / Wg) + 3):
                x0, y0 = ox + i * Wg, oy + i * Wg
                if x0 > maxx or y0 > maxy or x0 + 2 * Lg < minx or y0 + 2 * Lg < miny:
                    continue
                for r_ in (sbox(x0, y0, x0 + ln, y0 + w), sbox(x0, y0 + Wg, x0 + w, y0 + Wg + ln)):
                    if inner.contains(r_):
                        out.append(r_)
        return out

    best = []
    for px in np.linspace(0, Lg, 7, endpoint=False):
        for py in np.linspace(0, Wg, 4, endpoint=False):
            cand = layout(px, py)
            if len(cand) > len(best):
                best = cand
    rects = best
    n = 0
    mods, racks = [], []
    for rct in rects:
        r = affinity.rotate(rct, rot, origin=c)
        mb = r.minimum_rotated_rectangle
        # tilt about the low edge: model as slab rising across its short side
        cs = np.asarray(mb.exterior.coords)[:4]
        e0, e1 = cs[1] - cs[0], cs[2] - cs[1]
        if np.linalg.norm(e0) < np.linalg.norm(e1):
            cs = np.roll(cs, -1, axis=0)
            e0, e1 = cs[1] - cs[0], cs[2] - cs[1]
        rise = np.linalg.norm(e1) * math.sin(tilt)
        z0 = z_top + rack
        # build a tilted box via 8 points
        p = [np.r_[cs[0], z0], np.r_[cs[1], z0], np.r_[cs[2], z0 + rise], np.r_[cs[3], z0 + rise]]
        top = [q + np.array([0, 0, t]) for q in p]
        import trimesh
        verts = np.array(p + top)
        faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                 [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
        m = trimesh.Trimesh(verts, faces, process=True)
        m.fix_normals()
        mods.append(m)
        for k, q in enumerate(p):
            racks.append(G.box3(q[0] - 0.02, q[1] - 0.02, z_top, q[0] + 0.02, q[1] + 0.02, q[2]))
        n += 1
    model.add("roof", "pv/modules", "IfcSolarDevice", pv["material"], G.concat(mods), level="ROOF", bar=name,
              count=n)
    model.add("roof", "pv/racking", "IfcMember", "pv_frame", G.concat(racks), level="ROOF")
    model.flag(f"PV {name}: {n} modules ({n * 0.4:.1f} kW at ~400 W/module)")
