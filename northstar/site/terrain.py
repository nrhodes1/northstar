"""Terrain grading, ground surfaces (roads, sidewalks, lawns, lots), curbs,
road markings, and the landscape on our lot."""
from __future__ import annotations

import math

import numpy as np
import shapely
import trimesh
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box as sbox
from shapely.ops import unary_union

from .. import geom as G
from ..units import L, P
from .data import Raster, SiteData


# ---------------------------------------------------------------- grading
def graded(sd: SiteData, ctx, lot):
    """Return a Raster of finished grade (z rel. L1)."""
    sp = ctx.spec
    gr = sp["site"]["grading"]
    pad = ctx.outlines["L1"].buffer(gr["pad_buffer_m"], join_style=2)
    zpad = L(gr["pad_offset"])
    X, Y = sd.dem.xy()
    Z = sd.dem.z.copy()
    pts = shapely.points(X.ravel(), Y.ravel())
    inlot = shapely.contains(lot, pts).reshape(Z.shape)
    d_in = shapely.distance(pad, pts).reshape(Z.shape)
    d_out = shapely.distance(lot.boundary, pts).reshape(Z.shape)
    w = np.where(inlot, d_out / np.maximum(d_out + d_in, 1e-6), 0.0)
    w = np.clip(w, 0, 1)
    w = w * w * (3 - 2 * w)
    Z = Z * (1 - w) + zpad * w
    Z[shapely.contains(pad, pts).reshape(Z.shape)] = zpad
    return Raster(Z, sd.dem.x0, sd.dem.y0, sd.dem.res)


# ---------------------------------------------------------------- draping
def drape(poly, grade: Raster, offset=0.0, cell=None):
    """Triangulated surface of `poly` following the grade."""
    if poly is None or poly.is_empty:
        return None
    cell = cell or grade.res
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(math.floor(minx / cell) * cell, maxx + cell, cell)
    ys = np.arange(math.floor(miny / cell) * cell, maxy + cell, cell)
    if len(xs) < 2 or len(ys) < 2:
        return None
    X, Y = np.meshgrid(xs[:-1], ys[:-1])
    boxes = shapely.box(X.ravel(), Y.ravel(), X.ravel() + cell, Y.ravel() + cell)
    prepared = shapely.prepare(poly)
    inside = shapely.contains(poly, boxes)
    touch = shapely.intersects(poly, boxes) & ~inside
    verts, faces = [], []
    n = 0
    # full cells -> quads
    bx = X.ravel()[inside]
    by = Y.ravel()[inside]
    if len(bx):
        q = np.stack([np.c_[bx, by], np.c_[bx + cell, by], np.c_[bx + cell, by + cell], np.c_[bx, by + cell]], 1)
        q = q.reshape(-1, 2)
        verts.append(q)
        k = np.arange(len(bx)) * 4
        faces.append(np.c_[k, k + 1, k + 2])
        faces.append(np.c_[k, k + 2, k + 3])
        n = len(q)
    # partial cells -> clip + triangulate
    parts = shapely.intersection(boxes[touch], poly)
    for g in parts:
        for p in G.as_polys(g):
            if p.area < 1e-4:
                continue
            try:
                v, f = trimesh.creation.triangulate_polygon(p, engine="earcut")
            except Exception:
                continue
            verts.append(v)
            faces.append(f + n)
            n += len(v)
    if not verts:
        return None
    V = np.concatenate(verts)
    F = np.concatenate(faces)
    Z = grade.sample(V[:, 0], V[:, 1]) + offset
    m = trimesh.Trimesh(np.c_[V, Z], F, process=True)
    # make faces point up
    if m.face_normals[:, 2].mean() < 0:
        m.invert()
    return m


def strip(line: LineString, grade: Raster, width, z_lo, z_hi, step=1.0):
    """Vertical curb-like strip along a line (width across, from grade+z_lo
    to grade+z_hi). Left side of the line direction is the 'high' side."""
    if line.length < 0.2:
        return None
    n = max(2, int(line.length / step) + 1)
    pts = np.array([line.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n)])
    d = np.gradient(pts, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    nrm = np.c_[-d[:, 1], d[:, 0]]
    a = pts
    b = pts + nrm * width
    za = grade.sample(a[:, 0], a[:, 1])
    V = np.concatenate([np.c_[a, za + z_lo], np.c_[a, za + z_hi], np.c_[b, za + z_hi], np.c_[b, za + z_lo]])
    F = []
    for i in range(n - 1):
        for k in range(4):
            k2 = (k + 1) % 4
            p0, p1 = k * n + i, k2 * n + i
            F += [[p0, p0 + 1, p1 + 1], [p0, p1 + 1, p1]]
    m = trimesh.Trimesh(V, np.array(F), process=True)
    m.fix_normals()
    return m


# ---------------------------------------------------------------- classify
def vegetation_mask(sd: SiteData, sp):
    img = sd.ortho.img.astype(int)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    m = sp["site"]["surfaces"]["vegetation_margin"]
    veg = (g > r + m) & (g > b + m // 2)
    from scipy.ndimage import binary_opening, binary_closing
    veg = binary_closing(binary_opening(veg, iterations=2), iterations=2)
    return veg


def mask_to_polys(mask, x0, y0, res, min_area):
    import rasterio.features
    from rasterio.transform import Affine
    tr = Affine(res, 0, x0, 0, res, y0)
    out = []
    for geom, v in rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=tr):
        p = shapely.geometry.shape(geom)
        if p.area >= min_area:
            out.append(p.simplify(res * 0.8))
    return unary_union(out) if out else Polygon()


# ---------------------------------------------------------------- added streets
def added_streets(sp, disk, lot):
    """Streets missing from city GIS, from site.yaml `added_streets`."""
    rows = sp["site"].get("added_streets") or []
    if not rows:
        return None
    roads, bands, crossings, trees = [], [], [], []
    for st in rows:
        cl = LineString([P(p) for p in st["centerline"]])
        hw = L(st["width"]) / 2
        roads.append(cl.buffer(hw, cap_style=2))
        for side, sign in (("left", 1), ("right", -1)):
            off = hw
            for b in st.get(side, []):
                w = L(b["width"])
                o0 = cl.offset_curve(sign * off)
                o1 = cl.offset_curve(sign * (off + w))
                band = Polygon(list(o0.coords) + list(o1.coords)[::-1]).buffer(0)
                bands.append((b["kind"], band.difference(lot), b.get("material")))
                if b["kind"] == "planting_strip" and st.get("street_trees"):
                    t = st["street_trees"]
                    mid = cl.offset_curve(sign * (off + w / 2))
                    d = L(t.get("start_offset", 0))
                    while d < mid.length:
                        p = mid.interpolate(d)
                        trees.append({"x": p.x, "y": p.y, "h": t["height_m"], "species": t["species"],
                                      "r": t["height_m"] * t.get("crown_ratio", 0.3)})
                        d += L(t["spacing"])
                off += w
        if st.get("crosswalk_at_start"):
            crossings.append((cl, hw))
    road = unary_union(roads).intersection(disk)
    allg = unary_union([road] + [g for _, g, _ in bands])
    return {"road": road, "bands": bands, "all": allg, "crossings": crossings, "trees": trees}


# ---------------------------------------------------------------- surfaces
def build_ground(sd: SiteData, ctx, model, lot, footprints):
    sp = ctx.spec
    sf = sp["site"]["surfaces"]
    disk = Point(0, 0).buffer(sd.R, 64)
    grade = graded(sd, ctx, lot)
    parcels = unary_union([p for p, _ in sd.parcels if p.area > 1]).buffer(0)
    sidewalks = unary_union([p for p, _ in sd.sidewalks]).buffer(0).intersection(disk)
    row = disk.difference(parcels).difference(lot)
    # sidewalks are authoritative in the ROW; small slivers between parcel
    # lines and sidewalks become sidewalk too
    clean = lambda g: unary_union(G.as_polys(g)).buffer(0)
    road = clean(clean(row.difference(sidewalks.buffer(0.05))).buffer(-0.3).buffer(0.3))
    sidewalk = clean(clean(row.difference(road)).union(clean(sidewalks.intersection(disk))).difference(lot))
    raise_ = L(sf["sidewalk_raise"])
    extra = added_streets(sp, disk, lot)
    if extra:
        bands = unary_union([g for k, g, m in extra["bands"]])
        road = clean(road.union(extra["road"]).difference(bands))
        sidewalk = clean(sidewalk.difference(extra["road"]).difference(
            unary_union([g for k, g, m in extra["bands"] if k != "sidewalk"])).union(
            unary_union([g for k, g, m in extra["bands"] if k == "sidewalk"])))
        for k, g, m in extra["bands"]:
            if k == "planting_strip":
                model.add("site", "ground/planting_strip", "IfcGeographicElement", m, drape(g, grade, raise_ - 0.03))
        ctx._added_streets = extra
    model.add("site", "ground/road", "IfcCivilElement", sf["road_material"], drape(road, grade, -0.02))
    model.add("site", "ground/sidewalk", "IfcCivilElement", sf["sidewalk_material"],
              drape(sidewalk, grade, raise_ - 0.02))
    # curbs: sidewalk edge that faces the road
    edge = sidewalk.boundary.intersection(road.buffer(0.4))
    lines = [g for g in getattr(edge, "geoms", [edge]) if g.geom_type == "LineString"]
    cw = L(sf["curb_width"])
    curbs = []
    for ln in lines:
        # orient so that sidewalk is on the left
        mid = ln.interpolate(0.5, normalized=True)
        a = np.array(ln.coords[0]); b = np.array(ln.coords[-1])
        d = G.unit(b - a) if np.linalg.norm(b - a) > 1e-6 else np.array([1.0, 0])
        probe = Point(mid.x - d[1] * 0.5, mid.y + d[0] * 0.5)
        if not sidewalk.buffer(0.05).contains(probe):
            ln = LineString(list(ln.coords)[::-1])
        curbs.append(strip(ln, grade, cw, -0.25, raise_ - 0.01))
    model.add("site", "ground/curbs", "IfcCivilElement", sf["curb_material"], G.concat(curbs))

    # private parcels (not ours): lawn vs paved from the orthophoto
    veg = vegetation_mask(sd, sp)
    H, res = sd.ortho.H, sd.ortho.res
    vegp = mask_to_polys(veg, -H, -H, res, sf["min_patch_m2"])
    # ground continues under context buildings (views may hide them from camera)
    private = clean(parcels.intersection(disk).difference(lot))
    if extra:
        private = clean(private.difference(extra["all"]))
    lawn = clean(private.intersection(vegp))
    paved = clean(private.difference(lawn))
    model.add("site", "ground/lawn", "IfcGeographicElement", sf["lawn_material"], drape(lawn, grade, 0.0))
    model.add("site", "ground/paved", "IfcCivilElement", sf["paved_lot_material"], drape(paved, grade, 0.0))
    return grade, road, sidewalk


def build_markings(sd: SiteData, ctx, model, grade, road, sidewalk):
    sp = ctx.spec
    mk = sp["site"]["markings"]
    cl = mk["centerline"]
    w = L(cl["width"])
    gap = L(cl["double_gap"])
    strips = []
    for ln, props in sd.streets:
        if ln.geom_type == "MultiLineString":
            parts = list(ln.geoms)
        else:
            parts = [ln]
        name = (props.get("ST_NAME") or "") + " " + (props.get("ST_TYPE") or "")
        for part in parts:
            seg = part.intersection(road.buffer(-1.5))
            for s in getattr(seg, "geoms", [seg]):
                if s.geom_type != "LineString" or s.length < 4:
                    continue
                if "RIVER" not in name.upper() and props.get("CFCC", "A41")[:3] not in ("A31", "A21", "A35"):
                    continue
                for off in (-(gap / 2 + w / 2), gap / 2 + w / 2):
                    o = s.offset_curve(off)
                    if o.is_empty:
                        continue
                    strips.append(o.buffer(w / 2, cap_style=2))
    if strips:
        model.add("site", "markings/centerline", "IfcAnnotation", cl["material"],
                  drape(unary_union(strips).intersection(road), grade, 0.005, cell=2.0))
    # crosswalks at intersections near the site
    cw = mk["crosswalk"]
    ends = {}
    for ln, props in sd.streets:
        for part in getattr(ln, "geoms", [ln]):
            for c in (part.coords[0], part.coords[-1]):
                k = (round(c[0], 1), round(c[1], 1))
                ends.setdefault(k, []).append(part)
    bars = []
    for k, parts in ends.items():
        if len(parts) < 3 or math.hypot(*k) > 90:
            continue
        node = Point(k)
        for part in parts:
            # direction away from the node
            coords = list(part.coords)
            if Point(coords[-1]).distance(node) < 0.5:
                coords = coords[::-1]
            pl = LineString(coords)
            if pl.length < 15:
                continue
            # road half-width: distance to nearest sidewalk from a point 12 m out
            probe = pl.interpolate(12)
            hw = max(3.0, min(9.0, sidewalk.distance(probe)))
            dist = hw + cw["setback_m"]
            p0 = np.array(pl.interpolate(dist).coords[0])
            p1 = np.array(pl.interpolate(dist + 0.5).coords[0])
            d = G.unit(p1 - p0)
            nrm = np.array([-d[1], d[0]])
            depth = L(cw["depth"])
            bw, bg = L(cw["bar"]), L(cw["gap"])
            t = -hw
            while t < hw - bw:
                c = p0 + nrm * (t + bw / 2)
                bars.append(Polygon([c - nrm * bw / 2, c + nrm * bw / 2,
                                     c + nrm * bw / 2 + d * depth, c - nrm * bw / 2 + d * depth]))
                t += bw + bg
    extra = getattr(ctx, "_added_streets", None)
    if extra:
        bw, bg, depth = L(cw["bar"]), L(cw["gap"]), L(cw["depth"])
        for cl, hw in extra["crossings"]:
            a = np.array(cl.coords[0]); b = np.array(cl.coords[1])
            d = G.unit(b - a); nrm = np.array([-d[1], d[0]])
            p0 = a + d * 3.0
            t = -hw
            while t < hw - bw:
                c = p0 + nrm * (t + bw / 2)
                bars.append(Polygon([c - nrm * bw / 2, c + nrm * bw / 2,
                                     c + nrm * bw / 2 + d * depth, c - nrm * bw / 2 + d * depth]))
                t += bw + bg
    if bars:
        model.add("site", "markings/crosswalks", "IfcAnnotation", cw["material"],
                  drape(unary_union(bars).intersection(road), grade, 0.006, cell=2.0))


def build_lot_landscape(sd: SiteData, ctx, model, grade, lot, scatter):
    sp = ctx.spec
    ls = sp["site"]["landscape"]
    gl = sp["gallery"]
    rng = np.random.default_rng(7)
    bldg = ctx.outlines["L1"]
    hard = []
    if ls.get("porch_paving_from") == "gallery":
        deck = G.polygon([P(p) for p in gl["deck"]["outline"]])
        st = unary_union([G.polygon([P(p) for p in s["footprint"]]) for s in gl["stairs"]])
        hard.append(unary_union([deck, st]).buffer(0.6, join_style=2))
    pw = L(ls["path_width"])
    for pth in ls["paths"]:
        hard.append(LineString([P(p) for p in pth]).buffer(pw / 2, cap_style=2))
    # balconies at L1 (raised decks) need a paved landing
    hard = unary_union(hard).intersection(lot).difference(bldg)
    model.add("site", "landscape/paving", "IfcCivilElement", ls["hardscape_material"], drape(hard, grade, 0.03))
    plant = lot.difference(bldg.buffer(0.05)).difference(hard)
    model.add("site", "landscape/planting_beds", "IfcGeographicElement", ls["mulch_material"],
              drape(plant, grade, 0.02))
    # scatter shrubs / grasses + boulders + trees
    minx, miny, maxx, maxy = plant.bounds
    n = int(plant.area * ls["shrub_density_per_m2"])
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 20:
        tries += 1
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if plant.contains(Point(x, y)):
            pts.append((x, y))
    for x, y in pts:
        z = float(grade.sample(x, y))
        scatter["shrubs"].append({"x": x, "y": y, "z": z, "s": float(rng.uniform(0.35, 0.9)),
                                  "kind": int(rng.integers(0, 3))})
    for i in range(ls["boulders"]):
        for _ in range(50):
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if plant.buffer(-0.5).contains(Point(x, y)):
                scatter["boulders"].append({"x": x, "y": y, "z": float(grade.sample(x, y)),
                                            "s": float(rng.uniform(0.4, 0.9))})
                break
    extra = getattr(ctx, "_added_streets", None)
    for t in (extra or {}).get("trees", []):
        scatter["trees"].append({"x": t["x"], "y": t["y"], "z": float(grade.sample(t["x"], t["y"])),
                                 "h": t["h"], "r": t["r"], "species": t["species"], "src": "design"})
    for t in ls["trees"]:
        x, y = P(t["at"])
        h = t["height_m"]
        scatter["trees"].append({"x": x, "y": y, "z": float(grade.sample(x, y)), "h": h,
                                 "r": h * 0.33, "species": t["species"], "src": "design"})
