"""Exterior walls (layered assembly), openings, windows/doors, parapets."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from .. import geom as G
from ..model import Model
from ..units import FT, L, P, fmt_ftin
from .context import Ctx


# ------------------------------------------------------------------ helpers
def _host(ctx: Ctx, st: str, at):
    """Find host polygon edge for an opening at plan point `at` (m)."""
    pt = Point(at)
    best = None
    for _, poly in ctx.parts[st]:
        for a, b in G.edges(poly):
            seg = np.array([a, b])
            ab = b - a
            t = np.clip(np.dot(np.asarray(at) - a, ab) / np.dot(ab, ab), 0, 1)
            q = a + t * ab
            dist = np.linalg.norm(np.asarray(at) - q)
            if best is None or dist < best[0]:
                best = (dist, a, b, t * np.linalg.norm(ab))
    return best


def opening_list(ctx: Ctx):
    """Resolve openings.yaml into dicts with host edge + metres."""
    sp = ctx.spec
    types = sp["glazing"]["types"]
    out = []
    for st, rows in (sp.get("openings", {}).get("openings") or {}).items():
        for i, o in enumerate(rows or []):
            ty = types[o["type"]]
            at = P(o["at"])
            dist, a, b, u = _host(ctx, st, at)
            w = L(o["width"]) if "width" in o else L(ty["width"])
            raise_ = L(o.get("raise", 0))
            sill = L(o["sill"]) if "sill" in o else L(ty["sill"]) + raise_
            head = L(o["head"]) if "head" in o else L(ty["head"]) + raise_
            out.append(dict(
                id=f"{st}-{o['type']}-{i + 1:02d}", storey=st, type=o["type"], ty=ty,
                a=a, b=b, u=u, w=w, sill=sill, head=head, dist=dist,
            ))
    return out


def _cutter(ctx, o, z0, extra=0.0):
    d = G.unit(o["b"] - o["a"])
    return G.oriented_box(o["a"], d, o["u"] - o["w"] / 2 - extra, o["u"] + o["w"] / 2 + extra,
                          0.3, -(ctx.wall_T + 0.3), z0 + o["sill"] - extra, z0 + o["head"] + extra)


# ------------------------------------------------------------------ walls
def build_walls(ctx: Ctx, model: Model, openings):
    sp = ctx.spec
    env = sp["envelope"]
    par_h = L(env["parapet"]["height"])
    storeys = list(ctx.outlines)
    by_storey = {}
    for o in openings:
        by_storey.setdefault(o["storey"], []).append(o)

    for st in storeys:
        P_ = ctx.outlines[st]
        z0, z1, above = ctx.storey_span(st)
        st_above = ctx.storey_above(st)
        tf_here = 0.0 if st == storeys[0] else ctx.floor_T
        cutters = [_cutter(ctx, o, z0) for o in by_storey.get(st, [])]
        top_is_roof = st_above is None
        for name, d0, d1, mat, structural in ctx.layers:
            if mat is None:
                continue
            band = G.ring(P_, d0, d1)
            if name == "straw" or name == "plaster":
                zb, zt = z0, z1 - (ctx.roof_clt if top_is_roof else ctx.floor_T)
            else:   # cladding & sheathing wrap slab edge
                zb = z0 - (tf_here if st != storeys[0] else L("6\""))
                zt = z1 - (ctx.roof_clt if top_is_roof else ctx.floor_T)
            solid = G.extrude(band, zb, zt)
            solid = G.difference(solid, cutters)
            model.add("envelope", f"walls/{name}", "IfcCovering" if not structural else "IfcWall",
                      mat, solid, level=st, assembly_layer=name)
        # straw panel joints (annotation) + quantities
        _panel_layout(ctx, model, st, P_, z0, z1 - (ctx.roof_clt if top_is_roof else ctx.floor_T))

        # parapets / upstands on exposed edges (roof or terrace)
        exposed = P_ if top_is_roof else P_.difference(ctx.outlines[st_above].buffer(0.02, join_style=2))
        if exposed.area < 0.05:
            continue
        top = z1 - (ctx.roof_clt if top_is_roof else ctx.floor_T)
        if top_is_roof:
            ptop = z1 + par_h
        else:
            ptop = z1 + L(sp["roofs"]["terrace"]["curb_height"])
        clip = exposed.buffer(0.01, join_style=2)
        for name, d0, d1, mat, structural in ctx.layers:
            if mat is None:
                continue
            band = G.ring(P_, d0, d1).intersection(clip)
            # parapet core: straw doesn't continue; use framed parapet in its place
            m = mat if name not in ("straw",) else "wood_fibre_board"
            model.add("envelope", f"parapet/{name}", "IfcWall", m,
                      G.extrude(band, top, ptop), level=st)
        cov = env["parapet"]["coping_overhang"]
        band = G.ring(P_, -L(cov), ctx.wall_T + L(cov)).intersection(clip)
        model.add("envelope", "parapet/coping", "IfcCovering", env["parapet"]["coping_material"],
                  G.extrude(band, ptop, ptop + L("1 1/2\"")), level=st)


def _panel_layout(ctx, model, st, P_, zb, zt):
    pan = ctx.spec["envelope"]["panelization"]
    mod = L(pan["module_width"])
    min_w = L(pan.get("min_width", 0))
    max_h = L(pan["max_height"]) if pan.get("max_height") else None
    rows = 1
    if max_h and (zt - zb) > max_h:
        rows = math.ceil((zt - zb) / max_h)
    d0 = ctx.structure_face
    face = G.inset(P_, d0)
    n, short = 0, 0
    for poly in G.as_polys(face):
        for a, b in G.edges(poly):
            Lg = float(np.linalg.norm(b - a))
            k = max(1, math.ceil(Lg / mod - 1e-6))
            if Lg / k < min_w:
                short += 1
            n += k * rows
            d = G.unit(b - a)
            for i in range(1, k):
                p = a + d * (Lg * i / k)
                model.curve("envelope::panel_joints", [(p[0], p[1], zb), (p[0], p[1], zt)], storey=st)
            for r in range(1, rows):
                z = zb + (zt - zb) * r / rows
                model.curve("envelope::panel_joints", [(a[0], a[1], z), (b[0], b[1], z)], storey=st)
    msg = (f"Straw panels {st}: {n} panels (max width {fmt_ftin(mod)}), "
           f"{rows} row(s) of {fmt_ftin((zt - zb) / rows)}")
    if short:
        msg += f"; {short} wall run(s) narrower than {fmt_ftin(min_w)} -> site-built infill"
    model.flag(msg)


def build_base_flashing(ctx: Ctx, model: Model):
    base = ctx.spec["envelope"].get("base", {})
    if not base.get("drip_flashing"):
        return
    st = list(ctx.outlines)[0]
    P_ = ctx.outlines[st]
    proj = L(base.get("flashing_projection", "1\""))
    h = L(base.get("flashing_height", "2\""))
    clad_back = ctx.layers[0][2] + (ctx.layers[1][2] - ctx.layers[1][1] if ctx.layers[1][3] is None else 0)
    zb = ctx.spec.level_z(st) - L("6\"")
    band = G.ring(P_, -proj, clad_back)
    model.add("envelope", "base/drip_flashing", "IfcCovering", base.get("flashing_material", "coping_metal"),
              G.extrude(band, zb - 0.01, zb + h), level=st)


# ------------------------------------------------------------------ windows
def build_openings(ctx: Ctx, model: Model, openings):
    sp = ctx.spec
    gz = sp["glazing"]
    fd = gz["frame_defaults"]
    pl = gz["placement"]
    fw, fdep, sw = L(fd["frame_width"]), L(fd["frame_depth"]), L(fd["sash_width"])
    plaster = ctx.wall_T - ctx.structure_inner
    # frame position (depth from exterior face, positive inward)
    if pl["position"] == "interior":
        f_in = ctx.wall_T - plaster
    elif pl["position"] == "exterior":
        f_in = ctx.structure_face + fdep
    else:
        f_in = (ctx.structure_face + ctx.structure_inner + fdep) / 2
    f_in += L(pl.get("offset", 0))
    f_out = f_in - fdep
    lin = L(pl["reveal_lining"]["thickness"])
    lin_mat = pl["reveal_lining"]["material"]
    sf = pl["sill_flashing"]

    for o in openings:
        z0 = sp.level_z(o["storey"])
        ty = o["ty"]
        kind = ty.get("kind", "window")
        a, d = o["a"], G.unit(o["b"] - o["a"])
        u0, u1 = o["u"] - o["w"] / 2, o["u"] + o["w"] / 2
        zs, zh = z0 + o["sill"], z0 + o["head"]
        frame_mat = ty.get("frame_material", fd["frame_material"])
        box = lambda uu0, uu1, n0, n1, zz0, zz1: G.oriented_box(a, d, uu0, uu1, -n0, -n1, zz0, zz1)
        tag = dict(opening=o["id"], type=o["type"], storey=o["storey"],
                   width=fmt_ftin(o["w"]), sill=fmt_ftin(o["sill"]), head=fmt_ftin(o["head"]))

        # --- reveal linings (exterior side of frame)
        parts = [
            box(u0, u0 + lin, 0, f_out, zs, zh),            # jamb L
            box(u1 - lin, u1, 0, f_out, zs, zh),            # jamb R
            box(u0, u1, 0, f_out, zh - lin, zh),            # head
        ]
        model.add("glazing", "reveals", "IfcCovering", lin_mat, G.concat(parts), level=o["storey"], **tag)
        if o["sill"] > 0.05:
            # sloped metal sill: thin wedge approximated by tilted slab
            sill = box(u0 + lin, u1 - lin, -L(sf["projection"]), f_out, zs, zs + L("1/2\""))
            model.add("glazing", "sills", "IfcCovering", sf["material"], sill, level=o["storey"], **tag)
        # interior return (plaster) is already part of the cut wall.

        # --- frame
        ui0, ui1 = u0 + lin, u1 - lin
        zi0, zi1 = zs + (0 if kind in ("door", "slider") else 0), zh - lin
        if kind == "door":
            frame = G.concat([
                box(ui0, ui0 + fw, f_out, f_in, zi0, zi1),
                box(ui1 - fw, ui1, f_out, f_in, zi0, zi1),
                box(ui0, ui1, f_out, f_in, zi1 - fw, zi1),
            ])
            model.add("glazing", "frames", "IfcDoor", frame_mat, frame, level=o["storey"], **tag)
            leaf_t = L("1 3/4\"")
            nm = (f_out + f_in) / 2
            leaf = box(ui0 + fw, ui1 - fw, nm - leaf_t / 2, nm + leaf_t / 2, zi0, zi1 - fw)
            vl = ty.get("vision_lite")
            if vl:
                vw, vh = L(vl["width"]), L(vl["height"])
                uc = (ui0 + ui1) / 2
                cut = box(uc - vw / 2, uc + vw / 2, nm - 0.1, nm + 0.1, zi0 + L("2'-6\""), zi0 + L("2'-6\"") + vh)
                leaf = G.difference(leaf, [cut])
                glass = box(uc - vw / 2, uc + vw / 2, nm - L("1/4\""), nm + L("1/4\""),
                            zi0 + L("2'-6\""), zi0 + L("2'-6\"") + vh)
                model.add("glazing", "glass", "IfcDoor", gz["frame_defaults"]["glass_material"], glass,
                          level=o["storey"], **tag)
            model.add("glazing", "door_leaves", "IfcDoor", ty.get("leaf_material", "door_wood"), leaf,
                      level=o["storey"], **tag)
            continue

        panes = int(ty.get("panes", 1))
        if kind == "storefront":
            panes = max(1, round(o["w"] / L("5'-0\"")))
        bars = [
            box(ui0, ui0 + fw, f_out, f_in, zi0, zi1),
            box(ui1 - fw, ui1, f_out, f_in, zi0, zi1),
            box(ui0, ui1, f_out, f_in, zi1 - fw, zi1),
            box(ui0, ui1, f_out, f_in, zi0, zi0 + fw),
        ]
        pw = (ui1 - ui0 - 2 * fw) / panes
        for k in range(1, panes):
            uc = ui0 + fw + k * pw
            bars.append(box(uc - fw / 2, uc + fw / 2, f_out, f_in, zi0, zi1))
        if kind == "storefront" and (zi1 - zi0) > L("8'-0\""):
            zt = zi0 + L("7'-4\"")
            bars.append(box(ui0, ui1, f_out, f_in, zt - fw / 2, zt + fw / 2))
        model.add("glazing", "frames", "IfcWindow", frame_mat, G.concat(bars), level=o["storey"], **tag)
        # operable sashes (thin inner frames) for windows/sliders
        if kind in ("window", "slider"):
            sash = []
            for k in range(panes):
                s0 = ui0 + fw + k * pw + (fw / 2 if k else 0)
                s1 = ui0 + fw + (k + 1) * pw - (fw / 2 if k < panes - 1 else 0)
                sz0, sz1 = zi0 + fw, zi1 - fw
                nm0 = f_out + fdep * 0.25
                nm1 = f_out + fdep * 0.75
                sash += [box(s0, s0 + sw, nm0, nm1, sz0, sz1), box(s1 - sw, s1, nm0, nm1, sz0, sz1),
                         box(s0, s1, nm0, nm1, sz0, sz0 + sw), box(s0, s1, nm0, nm1, sz1 - sw, sz1)]
            model.add("glazing", "sashes", "IfcWindow", frame_mat, G.concat(sash), level=o["storey"], **tag)
        gm = (f_out + f_in) / 2
        glass = box(ui0 + fw, ui1 - fw, gm - L("1/2\""), gm + L("1/2\""), zi0 + fw, zi1 - fw)
        model.add("glazing", "glass", "IfcWindow", fd["glass_material"], glass, level=o["storey"], **tag)


def build_dressing(ctx: Ctx, model: Model, openings):
    """Sheer curtains behind a share of the windows (render dressing)."""
    import random
    dr = (ctx.spec["glazing"].get("dressing") or {}).get("curtains")
    if not dr:
        return
    rnd = random.Random(dr.get("seed", 1))
    gap = L(dr.get("gap", "3\""))
    inner = ctx.wall_T + gap
    parts = []
    for o in openings:
        if o["ty"].get("kind", "window") not in ("window", "slider") or rnd.random() > dr["fraction"]:
            continue
        z0 = ctx.spec.level_z(o["storey"])
        d = G.unit(o["b"] - o["a"])
        cover = rnd.uniform(dr.get("min_cover", 0.3), 1.0)
        zt = z0 + o["head"] + 0.05
        zb = zt - (o["head"] - o["sill"] + 0.1) * cover
        # one or two panels, partly drawn
        w = o["w"]
        if rnd.random() < 0.5:
            spans = [(o["u"] - w / 2 - 0.05, o["u"] - w / 2 + w * rnd.uniform(0.3, 0.6))]
        else:
            spans = [(o["u"] - w / 2 - 0.05, o["u"] + w / 2 + 0.05)]
        for u0, u1 in spans:
            parts.append(G.oriented_box(o["a"], d, u0, u1, -inner, -inner - 0.01, zb, zt))
    model.add("dressing", "curtains", "IfcFurnishingElement", dr["material"], G.concat(parts))
