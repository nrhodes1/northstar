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
        pitch = d.get("pitch", 0.068)
        depth = d.get("depth", 0.022)
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = 2 * math.pi / pitch
        axis = "X" if d.get("orientation", "vertical") == "vertical" else "Y"
        L_(sep.outputs[axis], mul.inputs[0])
        sin = nt.nodes.new("ShaderNodeMath")
        sin.operation = "SINE"
        L_(mul.outputs[0], sin.inputs[0])
        if d.get("profile") == "seam":
            pw = nt.nodes.new("ShaderNodeMath")
            pw.operation = "POWER"
            pw.inputs[1].default_value = 40
            L_(sin.outputs[0], pw.inputs[0])
            sin = pw
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 1.0
        bump.inputs["Distance"].default_value = depth
        L_(sin.outputs[0], bump.inputs["Height"])
        L_(bump.outputs[0], bsdf.inputs["Normal"])
        # subtle weathering/streak variation
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
    cam.clip_end = 3000
    ob = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(ob)
    pos = Vector(v["position"])
    tgt = Vector(v["target"])
    ob.location = pos
    d = tgt - pos
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    # 2-point perspective for street views: shift instead of tilt
    if v.get("two_point", pos.z < 3.0):
        hd = Vector((d.x, d.y, 0)).normalized()
        ob.rotation_euler = hd.to_track_quat("-Z", "Y").to_euler()
        horiz = math.hypot(d.x, d.y)
        cam.shift_y = (d.z / horiz) * cam.lens / cam.sensor_width * (16 / 9) * 0.5 if horiz else 0
    return ob


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
        names = list(views) if a.view == "all" else a.view.split(",")
        os.makedirs(os.path.join(out, "renders"), exist_ok=True)
        for n in names:
            bpy.context.scene.camera = cams[n]
            bpy.context.scene.render.filepath = os.path.join(out, "renders", f"{n}.png")
            bpy.ops.render.render(write_still=True)
            print("rendered", n)


if __name__ == "__main__":
    main()
