"""Context buildings rebuilt from Boston footprints + roof heights, with roof
forms and heights checked against the tiles surface model."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .. import geom as G
from .data import SiteData

FT = 0.3048


def _pick(lst, key):
    h = int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16)
    return lst[h % len(lst)]


def _samples(poly, raster, step=0.75):
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx + step / 2, maxx, step)
    ys = np.arange(miny + step / 2, maxy, step)
    if not len(xs) or not len(ys):
        c = poly.representative_point()
        return np.array([raster.sample(c.x, c.y)]), np.array([[c.x, c.y]])
    X, Y = np.meshgrid(xs, ys)
    pts = shapely.points(X.ravel(), Y.ravel())
    m = shapely.contains(poly, pts)
    P = np.c_[X.ravel()[m], Y.ravel()[m]]
    if not len(P):
        c = poly.representative_point()
        P = np.array([[c.x, c.y]])
    return raster.sample(P[:, 0], P[:, 1]), P


def _tent(rect: Polygon, z_eave, z_ridge, hip: bool):
    """Gable or hip roof volume over an oriented rectangle."""
    c = np.asarray(rect.exterior.coords)[:4]
    e0, e1 = c[1] - c[0], c[2] - c[1]
    if np.linalg.norm(e0) < np.linalg.norm(e1):
        c = np.roll(c, -1, axis=0)
        e0, e1 = c[1] - c[0], c[2] - c[1]
    Lg, W = np.linalg.norm(e0), np.linalg.norm(e1)
    u, v = e0 / Lg, e1 / W
    ov = 0.35  # eave overhang
    a = c[0] - u * ov - v * ov
    Lg2, W2 = Lg + 2 * ov, W + 2 * ov
    h = z_ridge - z_eave
    inset = min(W2 / 2, Lg2 / 2 - 0.5) if hip else 0.0
    r0 = a + v * W2 / 2 + u * inset
    r1 = a + v * W2 / 2 + u * (Lg2 - inset)
    base = [a, a + u * Lg2, a + u * Lg2 + v * W2, a + v * W2]
    ze = z_eave - 0.15
    V = [np.r_[p, ze] for p in base] + [np.r_[r0, z_ridge], np.r_[r1, z_ridge]]
    F = [[0, 2, 1], [0, 3, 2],          # bottom
         [0, 1, 5], [0, 5, 4],          # long side 1
         [2, 3, 4], [2, 4, 5],          # long side 2
         [1, 2, 5], [3, 0, 4]]          # ends (gable / hip)
    import trimesh
    m = trimesh.Trimesh(np.array(V), np.array(F), process=True)
    m.fix_normals()
    return m


def build_context(sd: SiteData, ctx, model, lot, grade):
    sp = ctx.spec
    cb = sp["context"]["buildings"]
    detail_r = sp["site"]["extent"]["detail_radius_m"]
    disk = Point(0, 0).buffer(sd.R)
    footprints = []
    count = {"from_gis": 0, "pitched": 0, "added_from_tiles": 0, "height_updated": 0}
    polys = []
    for g, props in sd.buildings:
        for p in G.as_polys(g):
            if p.area < 8 or not disk.contains(p.centroid):
                continue
            if p.intersection(lot).area > 0.3 * p.area:
                continue   # existing structures on our lot are replaced
            polys.append((p, props))

    for i, (p, props) in enumerate(polys):
        ground_ex = sd.dem.sample(*np.asarray(p.representative_point().coords[0]))
        zg, pts = _samples(p.buffer(-0.6) if p.buffer(-0.6).area > 4 else p, grade)
        base = float(np.min(zg)) - 0.2
        h2010 = float(props.get("BLDG_HGT_2010") or 0) * FT
        ds, _ = _samples(p.buffer(-0.8) if p.buffer(-0.8).area > 4 else p, sd.dsm)
        gnd = float(np.median(zg))
        rel = ds - gnd
        p10, p50, p90 = np.percentile(rel, [10, 50, 95])
        h = h2010 if h2010 > 2 else p90
        if p90 > 2.5 and abs(p90 - h) > 2.5 and h2010 > 0:
            h = p90
            count["height_updated"] += 1
        h = max(h, 3.0)
        use = str(props.get("PART_USE") or props.get("Land_Use") or "")
        typ = None
        for t in cb["typologies"]:
            if "min_height_m" in t and h < t["min_height_m"]:
                continue
            if "use" in t and not any(use.startswith(u) for u in t["use"]):
                continue
            typ = t
            break
        wall_mat = _pick(typ["material"], i)
        roof_mat = typ["roof"]
        rd = cb["roof_detection"]
        pitched = (typ["name"] == "house" and (p90 - p10) > rd["flat_if_range_below_m"]
                   and (p90 - p10) > rd["min_pitch_rise_m"] and p.area < 600)
        z_top = gnd + h
        if pitched:
            z_eave = gnd + max(p10, 3.0)
            z_eave = min(z_eave, z_top - 1.2)
        else:
            z_eave = z_top
        body = G.extrude(p, base, z_eave)
        model.add("context", f"buildings/walls", "IfcBuilding", wall_mat, body, typology=typ["name"],
                  height_m=round(h, 1))
        if pitched:
            rect = p.minimum_rotated_rectangle
            # hip if the DSM drops near both short ends
            hip = _is_hip(rect, sd.dsm, gnd, z_top)
            tent = _tent(rect, z_eave, z_top, hip)
            prism = G.extrude(p.buffer(0.35, join_style=2), z_eave - 0.2, z_top + 1)
            try:
                import trimesh
                roof = trimesh.boolean.intersection([tent, prism], engine="manifold")
            except Exception:
                roof = tent
            model.add("context", "buildings/roofs", "IfcRoof", roof_mat, roof)
            count["pitched"] += 1
        else:
            par = G.extrude(G.ring(p, 0, 0.25), z_eave, z_eave + 0.6)
            cap = G.extrude(p.buffer(-0.25), z_eave - 0.05, z_eave + 0.05)
            model.add("context", "buildings/roofs", "IfcRoof", roof_mat, G.concat([par, cap]))
        if cb["windows"]["enabled"] and p.centroid.distance(Point(0, 0)) < detail_r:
            _windows(model, p, base + 0.2, z_eave, cb, typ, gnd)
        footprints.append(p)
        count["from_gis"] += 1

    if cb["add_missing_from_tiles"]:
        added = _missing_from_tiles(sd, ctx, model, lot, unary_union(footprints), grade, cb)
        footprints += added
        count["added_from_tiles"] = len(added)
    model.flag("Context buildings: " + ", ".join(f"{k} {v}" for k, v in count.items()))
    return unary_union(footprints) if footprints else Polygon()


def _is_hip(rect, dsm, gnd, z_top):
    c = np.asarray(rect.exterior.coords)[:4]
    e0, e1 = c[1] - c[0], c[2] - c[1]
    if np.linalg.norm(e0) < np.linalg.norm(e1):
        c = np.roll(c, -1, axis=0)
        e0, e1 = c[1] - c[0], c[2] - c[1]
    mid_end0 = c[0] + e1 / 2 + G.unit(e0) * 0.8
    mid_end1 = c[1] + e1 / 2 - G.unit(e0) * 0.8
    centre = c[0] + e0 / 2 + e1 / 2
    zc = float(dsm.sample(*centre))
    ze = (float(dsm.sample(*mid_end0)) + float(dsm.sample(*mid_end1))) / 2
    return (zc - ze) > 0.45 * max(z_top - gnd - 3.0, 1.0)


def _windows(model, poly, z0, z1, cb, typ, gnd):
    if poly.geom_type != "Polygon":
        for p in G.as_polys(poly):
            _windows(model, p, z0, z1, cb, typ, gnd)
        return
    w = cb["windows"]
    fh = cb["floor_height_m"]
    floors = max(1, int((z1 - z0 - 0.4) / fh))
    glass, frame = [], []
    for a, b in G.edges(G.polygon(poly.exterior.coords)):
        Lg = float(np.linalg.norm(b - a))
        usable = Lg - 2 * w["edge_margin_m"]
        if usable < w["width_m"]:
            continue
        n = max(1, int(usable / w["spacing_m"]) + 1)
        d = G.unit(b - a)
        for k in range(n):
            u = w["edge_margin_m"] + (usable - w["width_m"]) * (k / max(n - 1, 1) if n > 1 else 0.5) \
                + w["width_m"] / 2
            for f in range(floors):
                zs = z0 + f * fh + w["sill_m"]
                zh = zs + w["height_m"]
                if zh > z1 - 0.3:
                    continue
                fr = w["frame_m"]
                frame.append(G.oriented_box(a, d, u - w["width_m"] / 2 - fr, u + w["width_m"] / 2 + fr,
                                            0.06, -0.02, zs - fr, zh + fr))
                glass.append(G.oriented_box(a, d, u - w["width_m"] / 2, u + w["width_m"] / 2,
                                            0.075, 0.0, zs, zh))
    model.add("context", "buildings/window_trim", "IfcCovering", "ctx_trim", G.concat(frame))
    model.add("context", "buildings/windows", "IfcWindow", "ctx_window", G.concat(glass))


def _rectilinear(poly, cell=1.5):
    """Square up a noisy blob: rotate to its dominant axis, rasterise on a
    `cell` grid, re-polygonise, rotate back."""
    from shapely import affinity
    import rasterio.features
    from rasterio.transform import Affine
    mrr = poly.minimum_rotated_rectangle
    c = np.asarray(mrr.exterior.coords)
    e = c[1] - c[0]
    ang = math.degrees(math.atan2(e[1], e[0]))
    org = poly.centroid
    r = affinity.rotate(poly, -ang, origin=org)
    minx, miny, maxx, maxy = r.bounds
    nx, ny = int((maxx - minx) / cell) + 2, int((maxy - miny) / cell) + 2
    tr = Affine(cell, 0, minx - cell / 2, 0, cell, miny - cell / 2)
    m = rasterio.features.rasterize([(r, 1)], out_shape=(ny, nx), transform=tr, all_touched=False)
    parts = [shapely.geometry.shape(g) for g, v in rasterio.features.shapes(m, mask=m > 0, transform=tr)]
    if not parts:
        return Polygon()
    u = unary_union(parts).simplify(0.01)
    return affinity.rotate(u, ang, origin=org)


def _missing_from_tiles(sd, ctx, model, lot, have, grade, cb):
    from .terrain import mask_to_polys, vegetation_mask
    H, res = sd.ortho.H, sd.ortho.res
    X, Y = np.meshgrid(-H + (np.arange(sd.ortho.top.shape[1]) + 0.5) * res,
                       -H + (np.arange(sd.ortho.top.shape[0]) + 0.5) * res)
    ground = sd.dem.sample(X.ravel(), Y.ravel()).reshape(X.shape)
    ndsm = sd.ortho.top - ground
    veg = vegetation_mask(sd, ctx.spec)
    from scipy.ndimage import binary_opening
    tall = binary_opening((ndsm > 4.0) & ~veg, iterations=3)
    blobs = mask_to_polys(tall, -H, -H, res, cb["missing_min_area_m2"])
    disk = Point(0, 0).buffer(sd.R)
    added = []
    exclude = unary_union([have.buffer(1.5), lot.buffer(1.0), ctx.outlines["L1"].buffer(3)])
    for p in G.as_polys(blobs):
        q = p.difference(exclude)
        for qq in G.as_polys(q):
            if qq.area < cb["missing_min_area_m2"] or not disk.contains(qq.centroid):
                continue
            qq = _rectilinear(qq, cell=1.5)
            if qq.is_empty:
                continue
            if qq.is_empty:
                continue
            zg, _ = _samples(qq, grade)
            ds, _ = _samples(qq.buffer(-1.0) if qq.buffer(-1.0).area > 4 else qq, sd.dsm)
            gnd = float(np.median(zg))
            h = float(np.percentile(ds - gnd, 60))
            if h < 4:
                continue
            typ = next(t for t in cb["typologies"] if h >= t.get("min_height_m", 0) and "use" not in t)
            body = G.extrude(qq, float(np.min(zg)) - 0.2, gnd + h)
            model.add("context", "buildings/walls", "IfcBuilding", _pick(typ["material"], qq.area), body,
                      typology=typ["name"] + "_from_tiles", height_m=round(h, 1))
            par = G.extrude(G.ring(qq, 0, 0.25), gnd + h, gnd + h + 0.8)
            model.add("context", "buildings/roofs", "IfcRoof", typ["roof"], par)
            _windows(model, qq, float(np.min(zg)), gnd + h, cb, typ, gnd)
            added.append(qq)
    return added
