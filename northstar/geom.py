"""Geometry kernel: shapely for 2D, trimesh + manifold3d for solids.

All coordinates in metres, drawing frame (see spec/project.yaml).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import shapely
import trimesh
from shapely.geometry import LineString, MultiPolygon, Polygon, box as sbox
from shapely.ops import unary_union

EPS = 1e-6


# --------------------------------------------------------------- 2D helpers
def polygon(points: Sequence[Sequence[float]]) -> Polygon:
    p = Polygon(points)
    if not p.is_valid:
        p = p.buffer(0)
    return shapely.geometry.polygon.orient(p, 1.0)


def as_polys(g) -> list[Polygon]:
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    if hasattr(g, "geoms"):
        out = []
        for x in g.geoms:
            out += as_polys(x)
        return out
    return []


def inset(p, d: float):
    """Mitred inward offset (negative d = outward)."""
    return p.buffer(-d, join_style=2, mitre_limit=10.0)


def ring(p, d0: float, d1: float):
    """Band between inward offsets d0 < d1 of p."""
    return inset(p, d0).difference(inset(p, d1))


def edges(p: Polygon):
    """Yield (a, b) exterior edges as numpy arrays, CCW polygon."""
    c = np.asarray(p.exterior.coords)[:-1]
    for i in range(len(c)):
        yield c[i], c[(i + 1) % len(c)]


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > EPS else v


def outward_normal(a, b):
    """Outward normal of CCW polygon edge a->b (right-hand side)."""
    d = unit(np.asarray(b) - np.asarray(a))
    return np.array([d[1], -d[0]])


# --------------------------------------------------------------- 3D solids
def extrude(g, z0: float, z1: float) -> trimesh.Trimesh | None:
    meshes = []
    for p in as_polys(g):
        if p.area < 1e-6:
            continue
        try:
            m = trimesh.creation.extrude_polygon(p, z1 - z0)
        except Exception:
            p = p.buffer(0)
            if p.is_empty:
                continue
            m = trimesh.creation.extrude_polygon(p, z1 - z0)
        m.apply_translation([0, 0, z0])
        meshes.append(m)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def oriented_box(origin2d, direction2d, u0, u1, n0, n1, z0, z1) -> trimesh.Trimesh:
    """Box aligned to a wall: u along `direction2d`, n along its left normal
    rotated to the right (outward for CCW edges): n = (d.y, -d.x)."""
    d = unit(direction2d)
    n = np.array([d[1], -d[0]])
    o = np.asarray(origin2d, float)
    corners = [o + d * u + n * w for u, w in ((u0, n0), (u1, n0), (u1, n1), (u0, n1))]
    poly = Polygon(corners)
    poly = shapely.geometry.polygon.orient(poly, 1.0)
    return extrude(poly, z0, z1)


def box3(xmin, ymin, zmin, xmax, ymax, zmax) -> trimesh.Trimesh:
    return extrude(sbox(xmin, ymin, xmax, ymax), zmin, zmax)


def segment_solid(a2, b2, width, z0, z1, extend=0.0):
    """Rectangular bar following segment a->b (plan), centred on it."""
    a2, b2 = np.asarray(a2, float), np.asarray(b2, float)
    d = unit(b2 - a2)
    L = float(np.linalg.norm(b2 - a2))
    return oriented_box(a2, d, -extend, L + extend, -width / 2, width / 2, z0, z1)


def bar3d(p0, p1, w, h) -> trimesh.Trimesh:
    """Square-ish bar between two 3D points (w across, h vertical-ish)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    Lg = np.linalg.norm(v)
    if Lg < EPS:
        return None
    m = trimesh.creation.box(extents=[Lg, w, h])
    x = v / Lg
    up = np.array([0, 0, 1.0])
    y = np.cross(up, x)
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0, 1.0, 0])
    y = unit(y)
    z = np.cross(x, y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
    T[:3, 3] = (p0 + p1) / 2
    m.apply_transform(T)
    return m


def cylinder_between(p0, p1, r, sections=10):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return trimesh.creation.cylinder(radius=r, segment=[p0, p1], sections=sections)


# --------------------------------------------------------------- booleans
def difference(a: trimesh.Trimesh, cutters: Iterable[trimesh.Trimesh]):
    cutters = [c for c in cutters if c is not None and len(c.faces)]
    if a is None or not cutters:
        return a
    try:
        return trimesh.boolean.difference([a] + cutters, engine="manifold")
    except Exception:
        out = a
        for c in cutters:
            try:
                out = trimesh.boolean.difference([out, c], engine="manifold")
            except Exception:
                pass
        return out


def union(meshes):
    meshes = [m for m in meshes if m is not None and len(m.faces)]
    if not meshes:
        return None
    try:
        return trimesh.boolean.union(meshes, engine="manifold")
    except Exception:
        return trimesh.util.concatenate(meshes)


def concat(meshes):
    meshes = [m for m in meshes if m is not None and len(m.faces)]
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def rotate_z(m, deg, about=(0, 0, 0)):
    T = trimesh.transformations.rotation_matrix(math.radians(deg), [0, 0, 1], about)
    m.apply_transform(T)
    return m
