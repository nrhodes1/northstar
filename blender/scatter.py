"""Vegetation + street entourage from out/site_scatter.json, using the free
Poly Haven models listed in spec/assets.yaml (cache/assets/models).

Falls back to simple procedural plants if the asset cache is missing.
"""
from __future__ import annotations

import json
import math
import os
import random

import bpy
from mathutils import Vector

LIB = None


def _lib():
    global LIB
    if LIB is None:
        LIB = bpy.data.collections.new("asset_library")
        bpy.context.scene.collection.children.link(LIB)
        LIB.hide_render = True
        LIB.hide_viewport = True
        lc = bpy.context.view_layer.layer_collection.children.get("asset_library")
        if lc:
            lc.exclude = True
    return LIB


def _append(asset_dir, mid):
    path = os.path.join(asset_dir, "models", mid, f"{mid}.blend")
    if not os.path.exists(path):
        return []
    with bpy.data.libraries.load(path) as (src, dst):
        dst.objects = list(src.objects)
    return [o for o in dst.objects if o is not None]


def _mw(o):
    """World matrix from transforms (appended objects aren't evaluated yet)."""
    m = o.matrix_basis.copy()
    p = o.parent
    while p is not None:
        m = p.matrix_basis @ o.matrix_parent_inverse @ m
        o, p = p, p.parent
    return m


def _base(name):
    return name.split(".")[0]


def _bbox(objs):
    mn = Vector((1e9, 1e9, 1e9))
    mx = Vector((-1e9, -1e9, -1e9))
    for o in objs:
        if o.type != "MESH":
            continue
        M = _mw(o)
        for c in o.bound_box:
            w = M @ Vector(c)
            mn = Vector((min(mn.x, w.x), min(mn.y, w.y), min(mn.z, w.z)))
            mx = Vector((max(mx.x, w.x), max(mx.y, w.y), max(mx.z, w.z)))
    return mn, mx


def tree_collections(asset_dir, mid, lod):
    """One collection per tree model (trunk + leaves at the given LOD)."""
    objs = _append(asset_dir, mid)
    if not objs:
        return []
    want = [o for o in objs if _base(o.name).endswith(f"_LOD{lod}") or _base(o.name).endswith("_trunk")]
    if not want:
        want = [o for o in objs if o.type == "MESH"]
    col = bpy.data.collections.new(f"lib_{mid}_LOD{lod}")
    _lib().children.link(col)
    for o in want:
        col.objects.link(o)
    mn, mx = _bbox(want)
    col.instance_offset = ((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z)
    col["height"] = mx.z - mn.z
    col["radius"] = max(mx.x - mn.x, mx.y - mn.y) / 2
    return [col]


def variant_collections(asset_dir, mid, lod_pref=(1, 0, 2), min_h=0.0):
    """One collection per plant variant (objects named *_<x>_LODn)."""
    objs = _append(asset_dir, mid)
    meshes = [o for o in objs if o.type == "MESH"]
    out = []
    for lod in lod_pref:
        sel = [o for o in meshes if _base(o.name).endswith(f"_LOD{lod}")]
        if sel:
            break
    else:
        sel = [o for o in meshes if "geonodes" not in o.name and "geometry" not in o.name] or meshes
    for o in sel:
        if "geonodes" in o.name:
            continue
        col = bpy.data.collections.new(f"lib_{o.name}")
        _lib().children.link(col)
        col.objects.link(o)
        mn, mx = _bbox([o])
        col.instance_offset = ((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z)
        col["height"] = mx.z - mn.z
        if col["height"] < min_h:
            continue
        out.append(col)
    return out


def instance(col, loc, rot_z=0.0, scale=1.0, name="inst", parent_col=None):
    e = bpy.data.objects.new(name, None)
    e.instance_type = "COLLECTION"
    e.instance_collection = col
    e.location = loc
    e.rotation_euler = (0, 0, rot_z)
    e.scale = (scale, scale, scale)
    (parent_col or bpy.context.scene.collection).objects.link(e)
    return e


def lawn_grass(cols, cam_xy, radius=60.0, density=420.0, seed=1):
    """Geometry-nodes grass on lawn / planting-strip ground near the camera."""
    targets = [o for o in bpy.data.objects if o.type == "MESH" and (
        o.name.startswith("site.ground.planting_strip"))]
    if not targets or not cols:
        return 0
    grass_col = bpy.data.collections.new("lib_lawn_variants")
    _lib().children.link(grass_col)
    for c in cols:
        grass_col.children.link(c)
    ng = bpy.data.node_groups.new("lawn_scatter", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N = ng.nodes
    L_ = ng.links.new
    gi = N.new("NodeGroupInput")
    go = N.new("NodeGroupOutput")
    dist = N.new("GeometryNodeDistributePointsOnFaces")
    dist.inputs["Density"].default_value = density
    dist.inputs["Seed"].default_value = seed
    pos = N.new("GeometryNodeInputPosition")
    vm = N.new("ShaderNodeVectorMath")
    vm.operation = "DISTANCE"
    vm.inputs[1].default_value = (cam_xy[0], cam_xy[1], 0)
    sep = N.new("ShaderNodeSeparateXYZ")
    comb = N.new("ShaderNodeCombineXYZ")
    L_(pos.outputs[0], sep.inputs[0])
    L_(sep.outputs["X"], comb.inputs["X"])
    L_(sep.outputs["Y"], comb.inputs["Y"])
    L_(comb.outputs[0], vm.inputs[0])
    cmp = N.new("FunctionNodeCompare")
    cmp.data_type = "FLOAT"
    cmp.operation = "GREATER_THAN"
    cmp.inputs["B"].default_value = radius
    L_(vm.outputs["Value"], cmp.inputs["A"])
    dele = N.new("GeometryNodeDeleteGeometry")
    ci = N.new("GeometryNodeCollectionInfo")
    ci.inputs["Collection"].default_value = grass_col
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    iop = N.new("GeometryNodeInstanceOnPoints")
    iop.inputs["Pick Instance"].default_value = True
    rnd = N.new("FunctionNodeRandomValue")
    rnd.data_type = "FLOAT_VECTOR"
    rnd.inputs["Min"].default_value = (0, 0, 0)
    rnd.inputs["Max"].default_value = (0, 0, 6.283)
    rs = N.new("FunctionNodeRandomValue")
    rs.data_type = "FLOAT"
    rs.inputs[2].default_value = 0.45
    rs.inputs[3].default_value = 0.9
    join = N.new("GeometryNodeJoinGeometry")
    L_(gi.outputs[0], dist.inputs["Mesh"])
    L_(dist.outputs["Points"], dele.inputs["Geometry"])
    L_(cmp.outputs["Result"], dele.inputs["Selection"])
    L_(dele.outputs["Geometry"], iop.inputs["Points"])
    L_(ci.outputs[0], iop.inputs["Instance"])
    L_(rnd.outputs["Value"], iop.inputs["Rotation"])
    L_(rs.outputs[1], iop.inputs["Scale"])
    L_(gi.outputs[0], join.inputs[0])
    L_(iop.outputs[0], join.inputs[0])
    L_(join.outputs[0], go.inputs[0])
    for o in targets:
        m = o.modifiers.new("lawn", "NODES")
        m.node_group = ng
    return len(targets)


def scatter(path, cfg):
    data = json.load(open(path))
    rnd = random.Random(3)
    asset_dir = cfg.get("assets_dir")
    models = cfg.get("models") or {}
    if not asset_dir or not os.path.exists(os.path.join(asset_dir, "models")):
        print("scatter: no asset cache - skipping vegetation")
        return
    veg = bpy.data.collections.new("vegetation")
    bpy.context.scene.collection.children.link(veg)
    near_trees = [c for m in models.get("trees_street", []) for c in tree_collections(asset_dir, m, 0)]
    ctx_trees_near = [c for m in models.get("trees_context", []) for c in tree_collections(asset_dir, m, 0)]
    ctx_trees_far = [c for m in models.get("trees_context", []) for c in tree_collections(asset_dir, m, 1)]
    shrubs = [c for m in models.get("shrubs", []) for c in variant_collections(asset_dir, m, min_h=0.25)]
    grasses = [c for m in models.get("grasses", []) for c in variant_collections(asset_dir, m, min_h=0.2)]
    ground = [c for m in models.get("groundcover", []) for c in variant_collections(asset_dir, m)]
    lawn = [c for m in models.get("lawn", []) for c in variant_collections(asset_dir, m, (0,))]
    boulders = [c for m in models.get("boulders", []) for c in variant_collections(asset_dir, m, (1, 0))]
    n_t = 0
    for t in data["trees"]:
        near = math.hypot(t["x"] - 14, t["y"] + 12) < 70
        if t.get("src") == "design" and near_trees:
            pool = near_trees
        else:
            pool = (ctx_trees_near if near else ctx_trees_far) or near_trees
        if not pool:
            continue
        col = pool[rnd.randrange(len(pool))]
        h = max(float(t["h"]), 3.0)
        s = h / max(col.get("height", 10.0), 0.5)
        instance(col, (t["x"], t["y"], t["z"] - 0.05), rnd.uniform(0, 6.283), s * rnd.uniform(0.92, 1.08),
                 name="tree_design" if t.get("src") == "design" else "tree_ctx", parent_col=veg)
        n_t += 1
    pools = {0: shrubs or grasses, 1: grasses or shrubs, 2: ground or grasses}
    for sh in data["shrubs"]:
        pool = pools.get(sh["kind"]) or []
        if not pool:
            continue
        col = pool[rnd.randrange(len(pool))]
        target = {0: 1.1, 1: 0.8, 2: 0.3}[sh["kind"]] * (0.6 + sh["s"])
        s = min(3.0 if sh["kind"] == 0 else 2.0, max(0.6, target / max(col.get("height", 0.5), 0.05)))
        instance(col, (sh["x"], sh["y"], sh["z"]), rnd.uniform(0, 6.283), s, name="plant", parent_col=veg)
    for b in data["boulders"]:
        if boulders:
            col = boulders[rnd.randrange(len(boulders))]
            s = 0.8 * b["s"] / max(col.get("height", 1.0), 0.1)
            instance(col, (b["x"], b["y"], b["z"] - 0.1), rnd.uniform(0, 6.283), s, name="boulder", parent_col=veg)
    # street furniture
    furn = data.get("furniture", [])
    cache = {}
    for f in furn:
        mid = (models.get(f["kind"]) or [None])[0]
        if not mid:
            continue
        if mid not in cache:
            objs = _append(asset_dir, mid)
            col = bpy.data.collections.new(f"lib_{mid}")
            _lib().children.link(col)
            for o in objs:
                if not o.name.endswith("_aged") and o.type in ("MESH", "EMPTY"):
                    col.objects.link(o)
            mn, mx = _bbox(list(col.objects))
            col.instance_offset = ((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z)
            cache[mid] = col
        instance(cache[mid], (f["x"], f["y"], f["z"]), f.get("rot", 0.0), f.get("scale", 1.0), name=f["kind"],
                 parent_col=veg)
    nl = 0
    lawn_pool = lawn or grasses
    for g in data.get("lawn", []):
        if not lawn_pool:
            break
        col = lawn_pool[rnd.randrange(len(lawn_pool))]
        s = min(1.6, max(0.6, 0.12 * g["s"] / max(col.get("height", 0.2), 0.02)))
        instance(col, (g["x"], g["y"], g["z"]), rnd.uniform(0, 6.283), s, name="lawn", parent_col=veg)
        nl += 1
    print(f"scatter: {n_t} trees, {len(data['shrubs'])} plants, {len(data['boulders'])} boulders, "
          f"{len(furn)} furniture, {nl} lawn tufts")
