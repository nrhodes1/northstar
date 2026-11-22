"""Assemble the Blender scene from build outputs and (optionally) render.

Run with Blender (4.2+):
  blender -b -P blender/build_scene.py -- --out out \
      [--site out/site.glb] [--view porch_street|all] [--samples 64] \
      [--res 1600x900] [--save out/northstar.blend]

Everything comes from out/: northstar_building.glb (+ .meta.json),
site.glb (terrain + context, optional), render_config.json (materials,
views and lighting resolved from spec/). Nothing in the .blend is hand-made:
re-running rebuilds it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta

import bpy
from mathutils import Vector

FT = 0.3048


# ------------------------------------------------------------------ args
def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--site", default=None)
    ap.add_argument("--view", default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--res", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--time", default=None, help="override HH:MM")
    ap.add_argument("--date", default=None)
    ap.add_argument("--cpu-threads", type=int, default=0)
    return ap.parse_known_args(argv)[0]


# ------------------------------------------------------------------ sun
def sun_position(lat, lon, dt_local, utc_offset_h):
    """NOAA solar position. Returns (azimuth_deg from north clockwise, elevation_deg)."""
    dt = dt_local - timedelta(hours=utc_offset_h)
    jd = (dt - datetime(2000, 1, 1, 12)).total_seconds() / 86400 + 2451545.0
    T = (jd - 2451545.0) / 36525
    L0 = (280.46646 + T * (36000.76983 + 0.0003032 * T)) % 360
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    Mr = math.radians(M)
    C = (math.sin(Mr) * (1.914602 - T * (0.004817 + 0.000014 * T)) + math.sin(2 * Mr) * (0.019993 - 0.000101 * T)
         + math.sin(3 * Mr) * 0.000289)
    true_long = L0 + C
    omega = 125.04 - 1934.136 * T
    lam = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    decl = math.degrees(math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(lam))))
    y = math.tan(math.radians(eps / 2)) ** 2
    L0r = math.radians(L0)
    eot = 4 * math.degrees(y * math.sin(2 * L0r) - 2 * e * math.sin(Mr) + 4 * e * y * math.sin(Mr) * math.cos(2 * L0r)
                           - 0.5 * y * y * math.sin(4 * L0r) - 1.25 * e * e * math.sin(2 * Mr))
    minutes = dt.hour * 60 + dt.minute + dt.second / 60
    tst = (minutes + eot + 4 * lon) % 1440
    ha = tst / 4 - 180
    latr, dr, har = math.radians(lat), math.radians(decl), math.radians(ha)
    cosz = math.sin(latr) * math.sin(dr) + math.cos(latr) * math.cos(dr) * math.cos(har)
    zen = math.degrees(math.acos(max(-1, min(1, cosz))))
    az = math.degrees(math.atan2(math.sin(har), math.cos(har) * math.sin(latr) - math.tan(dr) * math.cos(latr))) + 180
    return az % 360, 90 - zen


def us_eastern_offset(d):
    # DST: second Sunday in March -> first Sunday in November
    y = d.year
    mar = datetime(y, 3, 8)
    dst_start = mar + timedelta(days=(6 - mar.weekday()) % 7)
    nov = datetime(y, 11, 1)
    dst_end = nov + timedelta(days=(6 - nov.weekday()) % 7)
    return -4 if dst_start <= d < dst_end else -5


# ------------------------------------------------------------------ materials
def _nodes(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    return nt, bsdf, out


def _uv(nt):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["UV"], sep.inputs[0])
    return tc, sep


def _rib_bump(nt, bsdf, rib):
    """Cladding relief from envelope.yaml cladding_types.profile.
    sine = corrugated, seam = standing seam ridges, board = board joints."""
    L_ = nt.links.new
    tc, sep = _uv(nt)
    axis = "X" if rib.get("orientation", "vertical") == "vertical" else "Y"
    div = nt.nodes.new("ShaderNodeMath")
    div.operation = "DIVIDE"
    div.inputs[1].default_value = rib["pitch"]
    L_(sep.outputs[axis], div.inputs[0])
    fr = nt.nodes.new("ShaderNodeMath")
    fr.operation = "FRACT"
    L_(div.outputs[0], fr.inputs[0])
    shape = rib.get("shape", "sine")
    if shape == "board":
        h = nt.nodes.new("ShaderNodeMath")       # 1 on the board face, 0 in the joint
        h.operation = "GREATER_THAN"
        h.inputs[1].default_value = 0.05
        L_(fr.outputs[0], h.inputs[0])
    else:
        ang = nt.nodes.new("ShaderNodeMath")
        ang.operation = "MULTIPLY"
        ang.inputs[1].default_value = 2 * math.pi
        L_(fr.outputs[0], ang.inputs[0])
        h = nt.nodes.new("ShaderNodeMath")
        h.operation = "SINE"
        L_(ang.outputs[0], h.inputs[0])
        if shape == "seam":                       # narrow ridge once per pan
            pos = nt.nodes.new("ShaderNodeMath")
            pos.operation = "MAXIMUM"
            pos.inputs[1].default_value = 0.0
            L_(h.outputs[0], pos.inputs[0])
            pw = nt.nodes.new("ShaderNodeMath")
            pw.operation = "POWER"
            pw.inputs[1].default_value = 40
            L_(pos.outputs[0], pw.inputs[0])
            h = pw
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = rib["depth"]
    L_(h.outputs[0], bump.inputs["Height"])
    L_(bump.outputs[0], bsdf.inputs["Normal"])


def build_material(name, d):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt, bsdf, out = _nodes(mat)
    col = list(d.get("color", [0.7, 0.7, 0.7]))[:3] + [1]
    bsdf.inputs["Base Color"].default_value = col
    bsdf.inputs["Roughness"].default_value = d.get("roughness", 0.6)
    bsdf.inputs["Metallic"].default_value = d.get("metallic", 0.0)
    if d.get("coat"):
        bsdf.inputs["Coat Weight"].default_value = d["coat"]
    sh = d.get("shader", "principled")
    L_ = nt.links.new
    if sh == "glass":
        bsdf.inputs["Transmission Weight"].default_value = 1.0
        bsdf.inputs["IOR"].default_value = d.get("ior", 1.5)
        bsdf.inputs["Roughness"].default_value = d.get("roughness", 0.02)
        bsdf.inputs["Base Color"].default_value = col
    elif sh == "corrugated_metal":
        tc, sep = _uv(nt)
        # subtle weathering / streak variation in roughness
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 0.6
        L_(tc.outputs["UV"], noise.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = d.get("roughness", 0.35) - 0.08
        mr.inputs["To Max"].default_value = d.get("roughness", 0.35) + 0.12
        L_(noise.outputs["Fac"], mr.inputs["Value"])
        L_(mr.outputs[0], bsdf.inputs["Roughness"])
    elif sh == "procedural_wood":
        tc, sep = _uv(nt)
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "Y"
        wave.inputs["Scale"].default_value = 18
        wave.inputs["Distortion"].default_value = 6
        wave.inputs["Detail"].default_value = 3
        L_(tc.outputs["UV"], wave.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        c0 = [c * 0.82 for c in col[:3]] + [1]
        ramp.color_ramp.elements[0].color = c0
        ramp.color_ramp.elements[1].color = col
        L_(wave.outputs["Fac"], ramp.inputs[0])
        L_(ramp.outputs[0], bsdf.inputs["Base Color"])
    elif sh in ("sedum", "foliage", "grass"):
        tc = nt.nodes.new("ShaderNodeTexCoord")
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = {"sedum": 9, "foliage": 14, "grass": 6}[sh]
        noise.inputs["Detail"].default_value = 8
        L_(tc.outputs["Object"], noise.inputs["Vector"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = [c * 0.55 for c in col[:3]] + [1]
        ramp.color_ramp.elements[1].color = [min(1, c * 1.35) for c in col[:3]] + [1]
        if sh == "sedum":
            e = ramp.color_ramp.elements.new(0.82)
            e.color = (0.48, 0.30, 0.16, 1)   # reddish sedum patches
            ramp.color_ramp.elements[-1].position = 0.9
        L_(noise.outputs["Fac"], ramp.inputs[0])
        L_(ramp.outputs[0], bsdf.inputs["Base Color"])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.6
        L_(noise.outputs["Fac"], bump.inputs["Height"])
        L_(bump.outputs[0], bsdf.inputs["Normal"])
        if sh in ("foliage", "sedum"):
            # per-object variation (each tree / roof slightly different)
            oi = nt.nodes.new("ShaderNodeObjectInfo")
            hsv = nt.nodes.new("ShaderNodeHueSaturation")
            mr = nt.nodes.new("ShaderNodeMapRange")
            mr.inputs["To Min"].default_value = 0.47
            mr.inputs["To Max"].default_value = 0.54
            L_(oi.outputs["Random"], mr.inputs["Value"])
            L_(mr.outputs[0], hsv.inputs["Hue"])
            mr2 = nt.nodes.new("ShaderNodeMapRange")
            mr2.inputs["To Min"].default_value = 0.8
            mr2.inputs["To Max"].default_value = 1.15
            L_(oi.outputs["Random"], mr2.inputs["Value"])
            L_(mr2.outputs[0], hsv.inputs["Value"])
            L_(ramp.outputs[0], hsv.inputs["Color"])
            L_(hsv.outputs[0], bsdf.inputs["Base Color"])
        if sh == "foliage":
            bsdf.inputs["Subsurface Weight"].default_value = 0.15
            # leafy cut-out: voronoi cells on the clump surface become gaps
            vor = nt.nodes.new("ShaderNodeTexVoronoi")
            vor.feature = "DISTANCE_TO_EDGE"
            vor.inputs["Scale"].default_value = d.get("leaf_scale", 7.0)
            L_(tc.outputs["Object"], vor.inputs["Vector"])
            thr = nt.nodes.new("ShaderNodeMath")
            thr.operation = "GREATER_THAN"
            thr.inputs[1].default_value = d.get("leaf_gap", 0.06)
            L_(vor.outputs["Distance"], thr.inputs[0])
            tr = nt.nodes.new("ShaderNodeBsdfTransparent")
            mix = nt.nodes.new("ShaderNodeMixShader")
            L_(thr.outputs[0], mix.inputs["Fac"])
            L_(tr.outputs[0], mix.inputs[1])
            L_(bsdf.outputs[0], mix.inputs[2])
            L_(mix.outputs[0], out.inputs[0])
    elif sh in ("pavers", "asphalt"):
        tc = nt.nodes.new("ShaderNodeTexCoord")
        if sh == "pavers":
            brick = nt.nodes.new("ShaderNodeTexBrick")
            size = d.get("size", 0.6)
            brick.inputs["Scale"].default_value = 1.0 / size
            brick.inputs["Brick Width"].default_value = 1.0
            brick.inputs["Row Height"].default_value = 1.0
            brick.inputs["Mortar Size"].default_value = 0.006 / size
            brick.offset = 0.0
            brick.inputs["Color1"].default_value = col
            brick.inputs["Color2"].default_value = [c * 0.93 for c in col[:3]] + [1]
            brick.inputs["Mortar"].default_value = [c * 0.6 for c in col[:3]] + [1]
            L_(tc.outputs["UV"], brick.inputs["Vector"])
            L_(brick.outputs["Color"], bsdf.inputs["Base Color"])
            bump = nt.nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = 0.3
            L_(brick.outputs["Fac"], bump.inputs["Height"])
            L_(bump.outputs[0], bsdf.inputs["Normal"])
        else:
            noise = nt.nodes.new("ShaderNodeTexNoise")
            noise.inputs["Scale"].default_value = 40
            noise.inputs["Detail"].default_value = 10
            L_(tc.outputs["Object"], noise.inputs["Vector"])
            ramp = nt.nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].color = [c * 0.8 for c in col[:3]] + [1]
            ramp.color_ramp.elements[1].color = [c * 1.25 for c in col[:3]] + [1]
            L_(noise.outputs["Fac"], ramp.inputs[0])
            L_(ramp.outputs[0], bsdf.inputs["Base Color"])
    if d.get("rib"):
        _rib_bump(nt, bsdf, d["rib"])
    if d.get("brick"):
        tc, sep = _uv(nt)
        brick = nt.nodes.new("ShaderNodeTexBrick")
        brick.inputs["Scale"].default_value = 1.0
        brick.inputs["Brick Width"].default_value = 0.2
        brick.inputs["Row Height"].default_value = 0.068
        brick.inputs["Mortar Size"].default_value = 0.008
        brick.inputs["Color1"].default_value = col
        brick.inputs["Color2"].default_value = [c * 0.85 for c in col[:3]] + [1]
        brick.inputs["Mortar"].default_value = (0.6, 0.58, 0.55, 1)
        L_(tc.outputs["UV"], brick.inputs["Vector"])
        L_(brick.outputs["Color"], bsdf.inputs["Base Color"])
    if d.get("clapboard"):
        tc, sep = _uv(nt)
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 1 / 0.1016
        L_(sep.outputs["Y"], mul.inputs[0])
        fr = nt.nodes.new("ShaderNodeMath")
        fr.operation = "FRACT"
        L_(mul.outputs[0], fr.inputs[0])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Distance"].default_value = 0.01
        L_(fr.outputs[0], bump.inputs["Height"])
        L_(bump.outputs[0], bsdf.inputs["Normal"])
    return mat


def apply_materials(cfg):
    lib = cfg["materials"]
    for name, d in lib.items():
        build_material(name, d)
    for ob in bpy.data.objects:
        if ob.type != "MESH":
            continue
        for slot in ob.material_slots:
            if slot.material is None:
                continue
            base = slot.material.name.split(".")[0]
            if base in lib and slot.material.name != base:
                slot.material = bpy.data.materials[base]
            elif base in lib:
                build_material(base, lib[base])


# ------------------------------------------------------------------ scene
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path, collection):
    col = bpy.data.collections.new(collection)
    bpy.context.scene.collection.children.link(col)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    for ob in set(bpy.data.objects) - before:
        for c in ob.users_collection:
            c.objects.unlink(ob)
        col.objects.link(ob)
        if ob.type == "MESH":
            for p in ob.data.polygons:
                p.use_smooth = False
    return col


def setup_world(cfg, a):
    sc = bpy.context.scene
    lt = cfg["views"]["lighting"]
    date = a.date or lt["date"]
    tm = a.time or lt["time"]
    dt = datetime.strptime(f"{date} {tm}", "%Y-%m-%d %H:%M")
    az, el = sun_position(cfg["site"]["lat"], cfg["site"]["lon"], dt, us_eastern_offset(dt))
    # azimuth (from N, clockwise) -> true-north frame vector -> drawing frame
    azr, elr = math.radians(az), math.radians(el)
    v_site = Vector((math.sin(azr) * math.cos(elr), math.cos(azr) * math.cos(elr), math.sin(elr)))
    th = math.radians(-cfg["site"]["plan_rotation_deg"])
    v = Vector((math.cos(th) * v_site.x - math.sin(th) * v_site.y,
                math.sin(th) * v_site.x + math.cos(th) * v_site.y, v_site.z))
    print(f"sun az {az:.1f} el {el:.1f} (drawing-frame dir {tuple(round(c, 3) for c in v)})")
    world = bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = elr
    # Nishita sun_rotation measured from +Y? Blender: rotation about Z, 0 = +Y.
    sky.sun_rotation = math.atan2(v.x, v.y)
    sky.air_density = lt.get("air", 1.0)
    sky.dust_density = lt.get("dust", 1.0)
    sky.ozone_density = lt.get("ozone", 1.0)
    sky.sun_disc = True
    nt.links.new(sky.outputs[0], bg.inputs[0])
    bg.inputs[1].default_value = lt.get("sky_strength", 0.35)
    return v, el


def add_camera(name, v):
    cam = bpy.data.cameras.new(name)
    cam.lens = v.get("lens", 28)
    cam.sensor_width = 36
    cam.sensor_fit = "HORIZONTAL"
    cam.clip_end = 3000
    ob = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(ob)
    pos = Vector(v["position"])
    tgt = Vector(v["target"])
    ob.location = pos
    d = tgt - pos
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    if v.get("two_point", pos.z < 3.0):
        # level camera (verticals stay vertical); frame the target with lens shift
        hd = Vector((d.x, d.y, 0)).normalized()
        ob.rotation_euler = hd.to_track_quat("-Z", "Y").to_euler()
        horiz = math.hypot(d.x, d.y)
        # shift is in units of the sensor width (horizontal fit)
        cam.shift_y = (d.z / horiz) * cam.lens / cam.sensor_width if horiz else 0
    ob["view"] = dict(v)
    return ob


def clear_foreground(cam_ob, v, aspect):
    """Hide (from camera rays only) context objects that sit between the
    camera and the target, inside the horizontal field of view."""
    pos = Vector(v["position"])
    tgt = Vector(v["target"])
    fwd = Vector((tgt.x - pos.x, tgt.y - pos.y)).normalized()
    reach = (Vector((tgt.x, tgt.y)) - Vector((pos.x, pos.y))).length - v.get("clear_margin_m", 6.0)
    half = math.atan(18.0 / v.get("lens", 28)) + math.radians(4)
    hidden = 0
    for ob in bpy.data.objects:
        if ob.type != "MESH":
            continue
        is_ctx = ob.name.startswith("context.") or ob.name.startswith("tree_ctx")
        if not is_ctx:
            continue
        corners = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        inside_box = min(xs) <= pos.x <= max(xs) and min(ys) <= pos.y <= max(ys)
        block = inside_box
        if not block:
            for c in corners + [Vector(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0))]:
                r = Vector((c.x - pos.x, c.y - pos.y))
                dist = r.dot(fwd)
                if 0 < dist < reach and r.length > 1e-6:
                    ang = math.acos(max(-1, min(1, dist / r.length)))
                    if ang < half:
                        block = True
                        break
        if block:
            if v.get("clear_foreground") == "all":
                ob.hide_render = True          # gone entirely (no shadow either)
            else:
                ob.visible_camera = False      # still casts shadows / reflects
            hidden += 1
    print(f"clear_foreground ({v.get('clear_foreground')}): {hidden} objects hidden")


def place_on_ground(cam_ob, v):
    """Set camera z = ground under it + eye_height (ground = site ground surfaces)."""
    dg = bpy.context.evaluated_depsgraph_get()
    x, y = v["position"][0], v["position"][1]
    origin = Vector((x, y, 500.0))
    z = None
    for _ in range(40):
        hit, loc, nrm, idx, ob, mtx = bpy.context.scene.ray_cast(dg, origin, Vector((0, 0, -1)))
        if not hit:
            break
        if ob.name.startswith("site.ground") or ob.name.startswith("site.landscape"):
            z = loc.z
            break
        origin = loc - Vector((0, 0, 0.01))
    if z is None:
        print("place_on_ground: no ground found; keeping z")
        return
    eye = v["eye_height"]
    cam_ob.location.z = z + eye
    # re-aim the lens shift for the new height
    tgt = Vector(v["target"])
    d = tgt - cam_ob.location
    horiz = math.hypot(d.x, d.y)
    cam = cam_ob.data
    cam.shift_y = (d.z / horiz) * cam.lens / cam.sensor_width if horiz else 0
    print(f"camera {cam_ob.name}: ground {z:.2f} m, eye {cam_ob.location.z:.2f} m")


def restore_visibility():
    for ob in bpy.data.objects:
        if ob.type == "MESH":
            ob.visible_camera = True
            ob.hide_render = False


def render_settings(cfg, a):
    sc = bpy.context.scene
    r = cfg["views"]["render"]
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = a.samples or r["samples"]
    sc.cycles.use_denoising = r.get("denoise", True)
    sc.cycles.max_bounces = 8
    sc.cycles.transparent_max_bounces = 8
    if a.cpu_threads:
        sc.render.threads_mode = "FIXED"
        sc.render.threads = a.cpu_threads
    w, h = (map(int, a.res.split("x")) if a.res else r["resolution"])
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = r.get("look", "AgX - Medium High Contrast")
    except Exception:
        pass
    sc.view_settings.exposure = r.get("exposure", 0.0)
    sc.render.film_transparent = r.get("film_transparent", False)
    sc.render.image_settings.file_format = "PNG"


def main():
    a = args()
    out = os.path.abspath(a.out)
    with open(os.path.join(out, "render_config.json")) as f:
        cfg = json.load(f)
    reset()
    import_glb(os.path.join(out, "northstar_building.glb"), "building")
    site = a.site or os.path.join(out, "site.glb")
    if os.path.exists(site):
        import_glb(site, "site")
    apply_materials(cfg)
    sun_dir, el = setup_world(cfg, a)
    render_settings(cfg, a)
    views = cfg["views"]["views"]
    cams = {k: add_camera(k, v) for k, v in views.items()}
    if os.path.exists(os.path.join(out, "site_scatter.json")):
        import importlib.util
        spec = importlib.util.spec_from_file_location("scatter", os.path.join(os.path.dirname(__file__), "scatter.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.scatter(os.path.join(out, "site_scatter.json"), cfg)
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save))
    if a.view:
        main = cfg["views"]["render"].get("main_view")
        names = list(views) if a.view == "all" else [main if n == "main" else n for n in a.view.split(",")]
        os.makedirs(os.path.join(out, "renders"), exist_ok=True)
        for n in names:
            bpy.context.scene.camera = cams[n]
            v = views[n]
            restore_visibility()
            if v.get("eye_height") is not None:
                place_on_ground(cams[n], v)
            if v.get("clear_foreground"):
                clear_foreground(cams[n], v, None)
            sc = bpy.context.scene
            base_w = int(a.res.split("x")[0]) if a.res else cfg["views"]["render"]["resolution"][0]
            base_h = int(a.res.split("x")[1]) if a.res else cfg["views"]["render"]["resolution"][1]
            if v.get("aspect"):
                aw, ah = v["aspect"]
                sc.render.resolution_x, sc.render.resolution_y = base_w, int(round(base_w * ah / aw))
            else:
                sc.render.resolution_x, sc.render.resolution_y = base_w, base_h
            bpy.context.scene.render.filepath = os.path.join(out, "renders", f"{n}.png")
            bpy.ops.render.render(write_still=True)
            print("rendered", n)


if __name__ == "__main__":
    main()
