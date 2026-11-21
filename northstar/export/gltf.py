"""Write a Model to .glb for Blender: one node per element, named by id,
material named by materials.yaml key, world-scale box-projected UVs.

Frame: metres, drawing frame (Blender scene uses the same frame; the site
builder places terrain/context into it).
"""
from __future__ import annotations

import json

import numpy as np
import trimesh

from ..model import Model


Z_TO_Y = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)


def box_uv(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Unweld and assign world-space (metres) box-projected UVs.
    Vertical faces: u = distance along the wall (horizontal tangent), v = z.
    Horizontal faces: u = x, v = y."""
    m = mesh.copy()
    m.unmerge_vertices()
    fn = m.face_normals
    V = m.vertices
    F = m.faces
    uv = np.zeros((len(V), 2))
    for i, (f, n) in enumerate(zip(F, fn)):
        p = V[f]
        if abs(n[2]) > 0.7:
            uv[f] = p[:, :2]
        else:
            t = np.array([-n[1], n[0]])
            tn = np.linalg.norm(t)
            t = t / tn if tn > 1e-9 else np.array([1.0, 0.0])
            uv[f, 0] = p[:, :2] @ t
            uv[f, 1] = p[:, 2]
    m.visual = trimesh.visual.TextureVisuals(uv=uv)
    return m


def write(model: Model, path, materials=None, extra_meshes=None):
    mats = materials or {}
    scene = trimesh.Scene()
    cache = {}
    meta = {}

    def pbr(key):
        if key in cache:
            return cache[key]
        d = mats.get(key) or {}
        col = d.get("color", [0.7, 0.7, 0.7])
        m = trimesh.visual.material.PBRMaterial(
            name=key or "default",
            baseColorFactor=[*col[:3], 1.0],
            metallicFactor=float(d.get("metallic", 0.0)),
            roughnessFactor=float(d.get("roughness", 0.7)),
        )
        cache[key] = m
        return m

    for e in list(model.elements) + list(extra_meshes or []):
        mesh = box_uv(e.mesh)
        mesh.visual.material = pbr(e.material)
        # glTF is Y-up: (x, y, z) -> (x, z, -y); Blender's importer converts back.
        mesh.apply_transform(Z_TO_Y)
        scene.add_geometry(mesh, node_name=e.id, geom_name=e.id)
        meta[e.id] = {"system": e.system, "layer": e.layer, "ifc": e.ifc, "material": e.material,
                      "level": e.level, **{k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                                           for k, v in e.props.items()}}
    scene.export(str(path))
    with open(str(path).replace(".glb", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=0)
    return path
