"""Procedural trees, shrubs and boulders instanced from out/site_scatter.json.

Tree variants are generated once (trunk + branches + leaf clumps) and each
tree is a linked duplicate scaled to its surveyed / measured height.
"""
from __future__ import annotations

import json
import math
import random

import bmesh
import bpy
from mathutils import Matrix, Vector, noise


def _mat(name):
    return bpy.data.materials.get(name)


def _icosphere(bm, center, radius, subdiv=1, squash=1.0, jitter=0.0):
    ret = bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=radius)
    for v in ret["verts"]:
        n = noise.noise(v.co * 3.1 + center * 0.7) if jitter else 0.0
        v.co = Vector((v.co.x, v.co.y, v.co.z * squash)) * (1 + jitter * n) + center
    return ret["verts"]


def _cylinder(bm, p0, p1, r0, r1, segs=7):
    d = p1 - p0
    Lg = d.length
    ret = bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r0, radius2=r1, depth=Lg)
    rot = d.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
    M = Matrix.Translation((p0 + p1) / 2) @ rot
    bmesh.ops.transform(bm, matrix=M, verts=ret["verts"])
    return ret["verts"]


def make_tree_variant(idx, leaf_mat, bark_mat, seed, form="round"):
    """Unit tree: height 10 m. Returns mesh datablock."""
    rnd = random.Random(seed)
    h = 10.0
    bm_bark = bmesh.new()
    bm_leaf = bmesh.new()
    trunk_h = rnd.uniform(0.32, 0.45) * h
    lean = Vector((rnd.uniform(-0.3, 0.3), rnd.uniform(-0.3, 0.3), 0))
    top = Vector((0, 0, trunk_h)) + lean
    _cylinder(bm_bark, Vector((0, 0, -0.3)), top, 0.22, 0.14)
    crown_c = Vector((0, 0, trunk_h + (h - trunk_h) * 0.5)) + lean
    rx = (h - trunk_h) * (0.55 if form == "round" else 0.35)
    rz = (h - trunk_h) * 0.5
    # main branches
    tips = []
    for b in range(rnd.randint(4, 6)):
        a = rnd.uniform(0, 2 * math.pi)
        up = rnd.uniform(0.35, 0.8)
        tip = crown_c + Vector((math.cos(a) * rx * 0.65, math.sin(a) * rx * 0.65, (up - 0.4) * rz))
        _cylinder(bm_bark, top, tip, 0.09, 0.03, segs=5)
        tips.append(tip)
    # leaf clumps on an ellipsoid shell + near branch tips
    n = 240
    for k in range(n):
        if k < len(tips) * 6:
            base = tips[k % len(tips)]
            c = base + Vector((rnd.gauss(0, 0.5), rnd.gauss(0, 0.5), rnd.gauss(0, 0.4)))
        else:
            u = rnd.uniform(-1, 1)
            t = rnd.uniform(0, 2 * math.pi)
            rr = rnd.uniform(0.55, 1.0) ** 0.5
            s = math.sqrt(1 - u * u)
            c = crown_c + Vector((s * math.cos(t) * rx * rr, s * math.sin(t) * rx * rr, u * rz * rr))
        _icosphere(bm_leaf, c, rnd.uniform(0.45, 0.85) * (rx / 4.6), subdiv=2, squash=0.75, jitter=0.55)
    me = bpy.data.meshes.new(f"tree_v{idx}")
    bm = bmesh.new()
    for src, mi in ((bm_bark, 0), (bm_leaf, 1)):
        tmp = bpy.data.meshes.new("tmp")
        src.to_mesh(tmp)
        for p in tmp.polygons:
            p.material_index = mi
        bm.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
        src.free()
    bm.to_mesh(me)
    bm.free()
    me.materials.append(bark_mat)
    me.materials.append(leaf_mat)
    return me


def make_shrub(kind, mat_a, mat_b, seed):
    rnd = random.Random(seed)
    bm = bmesh.new()
    if kind == 0:      # rounded shrub
        for k in range(7):
            c = Vector((rnd.gauss(0, 0.25), rnd.gauss(0, 0.25), 0.35 + rnd.uniform(0, 0.25)))
            _icosphere(bm, c, rnd.uniform(0.25, 0.4), subdiv=2, squash=0.8, jitter=0.3)
    elif kind == 1:    # ornamental grass: thin cones fanning out
        for k in range(26):
            a = rnd.uniform(0, 2 * math.pi)
            tilt = rnd.uniform(0.05, 0.45)
            tip = Vector((math.cos(a) * tilt, math.sin(a) * tilt, rnd.uniform(0.55, 0.9)))
            _cylinder(bm, Vector((0, 0, 0)), tip, 0.03, 0.002, segs=3)
    else:              # perennial clump with flower heads
        for k in range(10):
            c = Vector((rnd.gauss(0, 0.22), rnd.gauss(0, 0.22), rnd.uniform(0.15, 0.45)))
            _icosphere(bm, c, rnd.uniform(0.1, 0.18), subdiv=1, jitter=0.2)
    me = bpy.data.meshes.new(f"shrub_{kind}")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat_a if kind != 2 else mat_b)
    return me


def scatter(path, cfg):
    data = json.load(open(path))
    col = bpy.data.collections.new("vegetation")
    bpy.context.scene.collection.children.link(col)
    lib = cfg["materials"]
    leaf = _mat("tree_leaves")
    bark = _mat("tree_bark")
    meadow = _mat("meadow_planting") or leaf
    # flower material
    flower = bpy.data.materials.new("perennial_flowers")
    flower.use_nodes = True
    flower.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.55, 0.35, 0.6, 1)
    variants = []
    forms = ["round", "round", "upright", "round", "upright", "round"]
    for i in range(6):
        variants.append(make_tree_variant(i, leaf, bark, seed=11 + i, form=forms[i]))
    rnd = random.Random(3)
    for t in data["trees"]:
        me = variants[rnd.randrange(len(variants))]
        ob = bpy.data.objects.new("tree_design" if t.get("src") == "design" else "tree_ctx", me)
        s = max(t["h"], 3.0) / 10.0
        sx = s * (t["r"] / (0.38 * max(t["h"], 3.0))) if t.get("r") else s
        ob.scale = (sx, sx, s)
        ob.rotation_euler = (0, 0, rnd.uniform(0, 2 * math.pi))
        ob.location = (t["x"], t["y"], t["z"])
        col.objects.link(ob)
    shrubs = [make_shrub(k, meadow, flower, seed=20 + k) for k in range(3)]
    for s in data["shrubs"]:
        ob = bpy.data.objects.new("shrub", shrubs[s["kind"]])
        ob.scale = (s["s"],) * 3
        ob.rotation_euler = (0, 0, rnd.uniform(0, 6.28))
        ob.location = (s["x"], s["y"], s["z"])
        col.objects.link(ob)
    bm = bmesh.new()
    _icosphere(bm, Vector((0, 0, 0.15)), 0.5, subdiv=3, squash=0.6, jitter=0.25)
    bme = bpy.data.meshes.new("boulder")
    bm.to_mesh(bme)
    bm.free()
    bme.materials.append(_mat("boulder"))
    for b in data["boulders"]:
        ob = bpy.data.objects.new("boulder", bme)
        ob.scale = (b["s"] * 1.3, b["s"], b["s"] * 0.8)
        ob.rotation_euler = (0, 0, rnd.uniform(0, 6.28))
        ob.location = (b["x"], b["y"], b["z"])
        col.objects.link(ob)
    print(f"scatter: {len(data['trees'])} trees, {len(data['shrubs'])} shrubs, {len(data['boulders'])} boulders")
