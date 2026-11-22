"""High-resolution context buildings.

Footprints/heights/roof forms come from city GIS + tiles (context.py). This
module adds what you see up close, using the Google tiles as reference:
  - facade colour + material class per building (sampled from tile textures)
  - recessed windows with frames, glass, meeting rails, sills, lintels
  - entries (doors, stoops), stacked porches (triple-deckers), balconies
    (apartment blocks), cornices, base courses, rooftop units, chimneys
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point

from .. import geom as G

CACHE = Path(__file__).resolve().parents[2] / "cache"
_S = None


def samples(spec):
    global _S
    if _S is None:
        p = CACHE / "facade_samples.npz"
        if not p.exists():
            _S = False
            return _S
        d = np.load(p)
        dz = (float(d["z_ref_mm"]) - spec["project"]["site"]["level1_elevation_mm"]) / 1000.0
        P = d["P"].astype(float)
        P[:, 2] += dz
        from scipy.spatial import cKDTree
        C = d["C"].astype(float)
        gain = C.mean(axis=0).mean() / C.mean(axis=0)      # gray-world white balance (tiles are blue-cast)
        C = np.clip(C * gain, 0, 255)
        _S = {"P": P, "N": d["N"].astype(float), "C": C, "tree": cKDTree(P[:, :2])}
    return _S


def _h(key, mod):
    return int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16) % mod


def facade_colour(S, a, b, n, z0, z1):
    """Median tile colour on the outside face of edge a-b."""
    if not S:
        return None, 0
    mid = (a + b) / 2
    Lg = np.linalg.norm(b - a)
    idx = S["tree"].query_ball_point(mid, Lg / 2 + 1.0)
    if not idx:
        return None, 0
    P, N, C = S["P"][idx], S["N"][idx], S["C"][idx]
    d = G.unit(b - a)
    rel = P[:, :2] - a
    along = rel @ d
    perp = rel @ n
    ok = ((along > 0.3) & (along < Lg - 0.3) & (np.abs(perp) < 1.2) & (N @ n > 0.6)
          & (P[:, 2] > z0 + 0.6) & (P[:, 2] < z1 - 0.3))
    if ok.sum() < 6:
        return None, int(ok.sum())
    return np.median(C[ok], axis=0) / 255.0, int(ok.sum())


def classify(rgb):
    r, g, b = rgb * 255
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    if r - g > 10 and r - b > 14 and lum < 175:
        return "brick"
    if b - r > 18 and lum < 200:
        return "panel"
    return "siding"


def street_facing(a, b, n, streets):
    mid = (a + b) / 2
    best = 1e9
    for ln in streets:
        best = min(best, ln.distance(Point(mid + n * 6)) - ln.distance(Point(mid)))
    return best < -2.0


def build(model, poly, base, eave, gnd, typ, streets, spec, pitched, key):
    S = samples(spec)
    hr = spec["context"]["buildings"].get("high_res", {})
    fh = spec["context"]["buildings"]["floor_height_m"]
    floors = max(1, int(round((eave - gnd - 0.2) / fh)))
    P = G.polygon(poly.exterior.coords)
    cols = []
    edges = []
    for a, b in G.edges(P):
        n = G.outward_normal(a, b)
        c, cnt = facade_colour(S, a, b, n, gnd, eave)
        edges.append((a, b, n, c))
        if c is not None:
            cols.append((c, np.linalg.norm(b - a)))
    if cols:
        w = np.array([l for _, l in cols])
        rgb = np.average(np.array([c for c, _ in cols]), axis=0, weights=w)
    else:
        rgb = None
    cls = classify(rgb) if rgb is not None else {"apartment": "brick", "tower": "panel"}.get(typ, "siding")
    if rgb is None:
        rgb = np.array({"brick": [0.45, 0.27, 0.2], "panel": [0.8, 0.8, 0.8], "siding": [0.82, 0.8, 0.74]}[cls])
    rgb = np.clip(rgb * 1.15, 0, 1)     # tiles are dark (baked shade) - lift toward albedo
    wall_mat = {"brick": "ctx_wall_brick", "panel": "ctx_wall_panel", "siding": "ctx_wall_siding"}[cls]
    trim = "ctx_trim" if cls != "brick" else "ctx_trim_dark"
    apartment = typ in ("apartment", "tower") or (cls == "brick" and floors >= 3)
    triple = cls != "brick" and ((typ == "house" and floors >= 3) or (typ == "house" and floors == 2 and _h(key, 3) == 0))

    W = hr.get("window", {})
    ww = W.get("width_m", 0.95 if not apartment else 1.2)
    wh = W.get("height_m", 1.55)
    sill_h = W.get("sill_m", 0.85)
    recess = W.get("recess_m", 0.14)
    spacing = W.get("spacing_m", 2.6 if not apartment else 3.0)
    margin = 0.9

    cutters, frames, glass, sills, lintels, doors, extras, rails = [], [], [], [], [], [], [], []
    for a, b, n, c in edges:
        Lg = float(np.linalg.norm(b - a))
        if Lg < 2.0:
            continue
        d = G.unit(b - a)
        front = street_facing(a, b, n, streets)
        usable = Lg - 2 * margin
        k = max(1, int(usable / spacing) + 1) if usable > ww else 0
        us = [margin + ww / 2 + (usable - ww) * (i / (k - 1) if k > 1 else 0.5) for i in range(k)]
        door_u = None
        if front and Lg > 3.0:
            door_u = us[len(us) // 2] if us else Lg / 2
        balcony_us = []
        if apartment and front and Lg > 14 and floors >= 3:
            balcony_us = [us[len(us) // 3], us[(2 * len(us)) // 3]] if len(us) >= 4 else []
        for f in range(floors):
            z0 = gnd + f * fh
            for u in us:
                is_door = f == 0 and door_u is not None and abs(u - door_u) < 1e-6
                is_bal = f >= 1 and u in balcony_us
                if is_door or is_bal:
                    w_, zs, zh = (1.0 if is_door else 1.6), z0 + 0.02, z0 + 2.2
                else:
                    w_, zs, zh = ww, z0 + sill_h, z0 + sill_h + wh
                if zh > eave - 0.35:
                    continue
                cutters.append(G.oriented_box(a, d, u - w_ / 2, u + w_ / 2, 0.05, -recess, zs, zh))
                fr = 0.06
                ring = [G.oriented_box(a, d, u - w_ / 2, u - w_ / 2 + fr, -recess + 0.07, -recess, zs, zh),
                        G.oriented_box(a, d, u + w_ / 2 - fr, u + w_ / 2, -recess + 0.07, -recess, zs, zh),
                        G.oriented_box(a, d, u - w_ / 2, u + w_ / 2, -recess + 0.07, -recess, zh - fr, zh),
                        G.oriented_box(a, d, u - w_ / 2, u + w_ / 2, -recess + 0.07, -recess, zs, zs + fr)]
                if not (is_door or is_bal):
                    zm = (zs + zh) / 2 + 0.05          # double-hung meeting rail
                    ring.append(G.oriented_box(a, d, u - w_ / 2, u + w_ / 2, -recess + 0.06, -recess + 0.01,
                                               zm - 0.03, zm + 0.03))
                    sills.append(G.oriented_box(a, d, u - w_ / 2 - 0.06, u + w_ / 2 + 0.06, 0.06, -0.02,
                                                zs - 0.06, zs))
                    if cls == "brick":
                        lintels.append(G.oriented_box(a, d, u - w_ / 2 - 0.08, u + w_ / 2 + 0.08, 0.012, -0.01,
                                                      zh, zh + 0.22))
                frames += ring
                if is_door:
                    doors.append(G.oriented_box(a, d, u - w_ / 2 + fr, u + w_ / 2 - fr, -recess + 0.05,
                                                -recess + 0.01, zs, zh - fr))
                    if not apartment:   # stoop
                        for s_ in range(3):
                            extras.append(("ctx_stoop", G.oriented_box(
                                a, d, u - 0.9, u + 0.9, 0.05 + (3 - s_) * 0.3, 0.0, gnd - 0.4, z0 - 0.18 * s_ + 0.02)))
                    else:               # entry canopy
                        extras.append(("ctx_trim_dark", G.oriented_box(a, d, u - 1.2, u + 1.2, 1.1, 0.0,
                                                                         zh + 0.25, zh + 0.38)))
                else:
                    glass.append(G.oriented_box(a, d, u - w_ / 2 + fr, u + w_ / 2 - fr, -recess + 0.04,
                                                -recess + 0.03, zs + fr, zh - fr))
                if is_bal:
                    bw = 2.4
                    extras.append(("ctx_trim_dark", G.oriented_box(a, d, u - bw / 2, u + bw / 2, 1.3, 0.0,
                                                                     z0 - 0.15, z0)))
                    for (u0, u1, n0, n1) in ((u - bw / 2, u + bw / 2, 1.3, 1.26), (u - bw / 2, u - bw / 2 + 0.04,
                                             1.3, 0.0), (u + bw / 2 - 0.04, u + bw / 2, 1.3, 0.0)):
                        rails.append(G.oriented_box(a, d, u0, u1, n0, n1, z0 + 0.95, z0 + 1.0))
                        nb = max(2, int((u1 - u0 if abs(u1 - u0) > 0.1 else abs(n0 - n1)) / 0.12))
                        for i in range(nb + 1):
                            t = i / nb
                            if abs(u1 - u0) > 0.1:
                                uu, nn = u0 + (u1 - u0) * t, (n0 + n1) / 2
                            else:
                                uu, nn = (u0 + u1) / 2, n0 + (n1 - n0) * t
                            rails.append(G.oriented_box(a, d, uu - 0.01, uu + 0.01, nn + 0.01, nn - 0.01,
                                                        z0, z0 + 0.98))
        # triple-decker porch stack on the street-facing short side
        if triple and front and 4.0 < Lg < 12:
            pw, pd = min(Lg * 0.65, 5.5), 2.2
            u0 = (Lg - pw) / 2
            for f in range(floors):
                z0 = gnd + f * fh
                extras.append(("ctx_porch", G.oriented_box(a, d, u0, u0 + pw, pd, 0.0, z0 + 0.25, z0 + 0.4)))
                for uu in (u0 + 0.1, u0 + pw - 0.1):
                    extras.append(("ctx_trim", G.oriented_box(a, d, uu - 0.09, uu + 0.09, pd - 0.02, pd - 0.2,
                                                               z0 + 0.4, z0 + fh + 0.25)))
                rails.append(G.oriented_box(a, d, u0, u0 + pw, pd, pd - 0.05, z0 + 1.2, z0 + 1.27))
                for i in range(int(pw / 0.14)):
                    uu = u0 + 0.07 + i * 0.14
                    rails.append(G.oriented_box(a, d, uu - 0.015, uu + 0.015, pd - 0.01, pd - 0.04,
                                                z0 + 0.4, z0 + 1.2))
            extras.append(("ctx_roof_flat", G.oriented_box(a, d, u0 - 0.1, u0 + pw + 0.1, pd + 0.2, 0.0,
                                                            gnd + floors * fh + 0.25, gnd + floors * fh + 0.4)))

    # walls with recessed openings
    walls = G.extrude(P, base, eave)
    walls = G.difference(walls, cutters)
    model.add("context", "buildings/walls", "IfcBuilding", wall_mat, walls, typology=typ, cls=cls,
              rgb=",".join(f"{v:.3f}" for v in rgb), height_m=round(eave - gnd, 1))
    # base course + cornice
    model.add("context", "buildings/base", "IfcCovering", "ctx_base",
              G.extrude(G.ring(P, -0.03, 0.0), base, gnd + 0.5))
    if not pitched:
        cor = 0.25 if cls == "brick" else 0.12
        model.add("context", "buildings/cornice", "IfcCovering", trim if cls != "brick" else "ctx_cornice",
                  G.extrude(G.ring(P, -cor, 0.2), eave - 0.35, eave + 0.05))
    else:
        model.add("context", "buildings/cornice", "IfcCovering", "ctx_trim",
                  G.extrude(G.ring(P, -0.2, 0.05), eave - 0.25, eave))
    model.add("context", "buildings/window_frames", "IfcWindow", trim, G.concat(frames))
    model.add("context", "buildings/windows", "IfcWindow", "ctx_window", G.concat(glass))
    model.add("context", "buildings/sills", "IfcCovering", "ctx_base" if cls == "brick" else trim, G.concat(sills))
    model.add("context", "buildings/lintels", "IfcCovering", "ctx_lintel", G.concat(lintels))
    model.add("context", "buildings/doors", "IfcDoor", "ctx_door", G.concat(doors))
    model.add("context", "buildings/railings", "IfcRailing", "ctx_rail", G.concat(rails))
    by = {}
    for m, mesh in extras:
        by.setdefault(m, []).append(mesh)
    for m, meshes in by.items():
        model.add("context", "buildings/details", "IfcBuildingElementProxy", m, G.concat(meshes))
    # roof furniture
    if not pitched and P.area > 120:
        c = np.array(P.representative_point().coords[0])
        units = []
        for i in range(1 + _h(key, 3)):
            off = np.array([(i - 1) * 3.2, (_h((key, i), 5) - 2) * 1.2])
            q = c + off
            if P.buffer(-1.5).contains(Point(q)):
                units.append(G.box3(q[0] - 1.0, q[1] - 0.7, eave + 0.05, q[0] + 1.0, q[1] + 0.7, eave + 1.2))
        model.add("context", "buildings/rooftop_units", "IfcBuildingElementProxy", "ctx_rtu", G.concat(units))
    if pitched and _h(key, 2) == 0:
        c = np.array(P.representative_point().coords[0])
        model.add("context", "buildings/chimney", "IfcBuildingElementProxy", "ctx_wall_brick",
                  G.box3(c[0] - 0.35, c[1] - 0.35, eave, c[0] + 0.35, c[1] + 0.35, eave + 3.2),
                  rgb="0.45,0.27,0.2")
    return cls
