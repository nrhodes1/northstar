"""Write a Model to .3dm (rhino3dm), layered by system, with metadata.

Geometry is written in millimetres in the client's site-model frame by
default, so the file drops straight onto the Google-tiles site model.
"""
from __future__ import annotations

import math

import numpy as np
import rhino3dm as r

from ..model import Model


def _xf(spec, frame):
    if frame != "site":
        return lambda p: (p[0] * 1000, p[1] * 1000, p[2] * 1000)
    s = spec["project"]["site"]
    th = math.radians(s["plan_rotation_deg"])
    c, si = math.cos(th), math.sin(th)
    ox, oy = s["plan_origin_mm"]
    oz = s["level1_elevation_mm"]

    def f(p):
        x, y, z = p[0] * 1000, p[1] * 1000, p[2] * 1000
        return (c * x - si * y + ox, si * x + c * y + oy, z + oz)
    return f


def write(model: Model, spec, path, frame="site", materials=None):
    f3 = r.File3dm()
    f3.Settings.ModelUnitSystem = r.UnitSystem.Millimeters
    if frame == "site":
        s = spec["project"]["site"]
        ea = f3.Settings.EarthAnchorPoint
        ea.EarthBasepointLatitude = s["anchor_lat"]
        ea.EarthBasepointLongitude = s["anchor_lon"]
        f3.Settings.EarthAnchorPoint = ea
    xf = _xf(spec, frame)
    mats = materials or {}
    mat_index = {}

    def material(key):
        if key in mat_index:
            return mat_index[key]
        m = r.Material()
        m.Name = key or "default"
        col = (mats.get(key) or {}).get("color", [0.7, 0.7, 0.7])
        m.DiffuseColor = tuple(int(255 * c) for c in col[:3]) + (255,)
        mat_index[key] = f3.Materials.Add(m)
        return mat_index[key]

    layer_index = {}

    def layer(path, color=None):
        if path in layer_index:
            return layer_index[path]
        parts = path.split("::")
        parent = None
        for i in range(len(parts)):
            sub = "::".join(parts[: i + 1])
            if sub in layer_index:
                parent = layer_index[sub]
                continue
            ly = r.Layer()
            ly.Name = parts[i]
            if parent is not None:
                ly.ParentLayerId = f3.Layers.FindIndex(parent).Id
            if color is not None and i == len(parts) - 1:
                ly.Color = color
            idx = f3.Layers.Add(ly)
            layer_index[sub] = idx
            parent = idx
        return layer_index[path]

    for e in model.elements:
        mi = material(e.material)
        col = (mats.get(e.material) or {}).get("color", [0.7, 0.7, 0.7])
        li = layer(f"{e.system}::{e.layer.replace('/', '::')}", tuple(int(255 * c) for c in col[:3]) + (255,))
        rm = r.Mesh()
        V = e.mesh.vertices
        for v in V:
            rm.Vertices.Add(*xf(v))
        for a, b, c in e.mesh.faces:
            rm.Faces.AddFace(int(a), int(b), int(c))
        rm.Normals.ComputeNormals()
        rm.Compact()
        att = r.ObjectAttributes()
        att.LayerIndex = li
        att.MaterialIndex = mi
        att.MaterialSource = r.ObjectMaterialSource.MaterialFromObject
        att.ColorSource = r.ObjectColorSource.ColorFromLayer
        att.Name = e.id
        att.SetUserString("id", e.id)
        att.SetUserString("ifc", e.ifc)
        att.SetUserString("material", e.material or "")
        if e.level:
            att.SetUserString("level", e.level)
        for k, v in e.props.items():
            att.SetUserString(str(k), str(v))
        f3.Objects.AddMesh(rm, att)

    for c in model.curves:
        li = layer(c["layer"], (200, 60, 60, 255))
        pts = [r.Point3d(*xf(p)) for p in c["points"]]
        pl = r.Polyline(len(pts))
        for p in pts:
            pl.Add(p.X, p.Y, p.Z)
        att = r.ObjectAttributes()
        att.LayerIndex = li
        f3.Objects.AddPolyline(pl, att)
    f3.Write(str(path), 8)
    return path
