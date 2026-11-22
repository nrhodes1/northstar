"""Photoreal layer for the Blender scene: image-based materials, HDRI sky +
sun lamp, depth of field, compositor post, interior lights.

All parameters come from render_config.json (spec/materials.yaml `tex`,
spec/views.yaml `lighting` / `render.post` / per-view keys, spec/assets.yaml).
"""
from __future__ import annotations

import math
import os

import bpy
from mathutils import Euler, Vector

ASSETS = None


def init(cfg):
    global ASSETS
    ASSETS = cfg.get("assets_dir")


# ---------------------------------------------------------------- textures
def _img(path, colorspace):
    if not os.path.exists(path):
        return None
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = colorspace
    return im


def apply_texture(mat, d):
    t = d.get("tex")
    if not t or not ASSETS:
        return
    folder = os.path.join(ASSETS, "textures", t["id"])
    diff = _img(os.path.join(folder, "diff.jpg"), "sRGB")
    nor = _img(os.path.join(folder, "nor.jpg"), "Non-Color")
    rough = _img(os.path.join(folder, "rough.jpg"), "Non-Color")
    if diff is None:
        print("missing texture", t["id"])
        return
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    L_ = nt.links.new
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    s = 1.0 / float(t.get("size", 2.0))
    mp.inputs["Scale"].default_value = (s, s, s)
    L_(tc.outputs["UV"], mp.inputs["Vector"])

    def tex(im):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = im
        n.interpolation = "Cubic"
        L_(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    mode = t.get("mode", "full")
    col = list(d.get("color", [0.7, 0.7, 0.7]))[:3] + [1]
    if mode in ("full", "tint"):
        dn = tex(diff)
        if mode == "full":
            L_(dn.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            # keep the spec colour, borrow the photo's variation (normalised luminance)
            bw = nt.nodes.new("ShaderNodeRGBToBW")
            L_(dn.outputs["Color"], bw.inputs["Color"])
            mr = nt.nodes.new("ShaderNodeMapRange")
            mr.inputs["From Min"].default_value = 0.15
            mr.inputs["From Max"].default_value = 0.75
            mr.inputs["To Min"].default_value = 0.7
            mr.inputs["To Max"].default_value = 1.2
            L_(bw.outputs["Val"], mr.inputs["Value"])
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 1.0
            mix.inputs[6].default_value = col
            L_(mr.outputs[0], mix.inputs[7])
            L_(mix.outputs[2], bsdf.inputs["Base Color"])
    if rough is not None:
        rn = tex(rough)
        rr = nt.nodes.new("ShaderNodeMapRange")
        base = d.get("roughness", 0.6)
        rr.inputs["To Min"].default_value = max(0.02, base - 0.25)
        rr.inputs["To Max"].default_value = min(1.0, base + 0.2)
        L_(rn.outputs["Color"], rr.inputs["Value"])
        L_(rr.outputs[0], bsdf.inputs["Roughness"])
    if nor is not None:
        nn = tex(nor)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = 0.6 if mode == "detail" else 1.0
        L_(nn.outputs["Color"], nm.inputs["Color"])
        # keep an existing rib bump (cladding) by feeding the map into it
        existing = [l for l in nt.links if l.to_socket == bsdf.inputs["Normal"]]
        if existing and existing[0].from_node.type == "BUMP":
            L_(nm.outputs["Normal"], existing[0].from_node.inputs["Normal"])
        else:
            L_(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if d.get("translucent"):
        bsdf.inputs["Transmission Weight"].default_value = 0.0
        bsdf.inputs["Subsurface Weight"].default_value = d["translucent"]


def object_colour(mat):
    """Multiply Base Color by the object's colour (set per building from tiles)."""
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    sock = bsdf.inputs["Base Color"]
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    if sock.is_linked:
        src = sock.links[0].from_socket
        nt.links.new(src, mix.inputs[6])
    else:
        mix.inputs[6].default_value = sock.default_value[:]
    nt.links.new(oi.outputs["Color"], mix.inputs[7])
    nt.links.new(mix.outputs[2], sock)


def set_object_colours(meta_path):
    import json
    if not os.path.exists(meta_path):
        return
    meta = json.load(open(meta_path))
    n = 0
    for name, m in meta.items():
        rgb = m.get("rgb")
        ob = bpy.data.objects.get(name)
        if rgb and ob:
            ob.color = tuple(float(v) for v in rgb.split(",")) + (1.0,)
            n += 1
    print(f"object colours from tiles: {n}")


def apply_textures(cfg):
    for name, d in cfg["materials"].items():
        m = bpy.data.materials.get(name)
        if m and d.get("tex"):
            apply_texture(m, d)
        if m and d.get("object_color"):
            object_colour(m)
        if m and d.get("translucent") and not d.get("tex"):
            bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            bsdf.inputs["Subsurface Weight"].default_value = d["translucent"]


# ---------------------------------------------------------------- lighting
def setup_hdri(cfg):
    lt = cfg["views"]["lighting"]
    path = cfg.get("hdri_path")
    if not path or not os.path.exists(path):
        return False
    world = bpy.context.scene.world
    nt = world.node_tree
    for n in list(nt.nodes):
        if n.type not in ("OUTPUT_WORLD",):
            nt.nodes.remove(n)
    out = next(n for n in nt.nodes if n.type == "OUTPUT_WORLD")
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(path, check_existing=True)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0, 0, math.radians(lt.get("hdri_rotation_deg", 0)))
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = lt.get("hdri_strength", 1.0)
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    # key light
    sl = lt.get("sun_lamp", {})
    sun = bpy.data.lights.new("sun_key", "SUN")
    sun.energy = sl.get("strength", 3.0)
    sun.angle = math.radians(sl.get("angle_deg", 1.0))
    sun.color = blackbody(sl.get("temperature_k", 5000))
    ob = bpy.data.objects.new("sun_key", sun)
    bpy.context.scene.collection.objects.link(ob)
    return True


def blackbody(k):
    """Approximate sRGB of a blackbody (K) - Tanner Helland fit."""
    t = k / 100.0
    r = 255 if t <= 66 else 329.7 * ((t - 60) ** -0.1332)
    g = 99.47 * math.log(t) - 161.12 if t <= 66 else 288.12 * ((t - 60) ** -0.0755)
    b = 255 if t >= 66 else (0 if t <= 19 else 138.52 * math.log(t - 10) - 305.04)
    c = [max(0, min(255, v)) / 255 for v in (r, g, b)]
    return tuple(x ** 2.2 for x in c)


def aim_sun(az, el, plan_rotation_deg):
    ob = bpy.data.objects.get("sun_key")
    if not ob:
        return
    azr, elr = math.radians(az), math.radians(el)
    vx, vy = math.sin(azr), math.cos(azr)
    th = math.radians(-plan_rotation_deg)
    d = Vector((math.cos(th) * vx - math.sin(th) * vy, math.sin(th) * vx + math.cos(th) * vy, 0))
    d = Vector((d.x * math.cos(elr), d.y * math.cos(elr), math.sin(elr)))   # toward the sun
    ob.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()


# ---------------------------------------------------------------- camera/post
def camera_extras(cam_ob, v, cfg):
    cam = cam_ob.data
    if v.get("shift"):
        cam.shift_x, cam.shift_y = v["shift"]
    dof = cfg["views"]["render"].get("dof")
    if dof and not v.get("ortho"):
        cam.dof.use_dof = True
        tgt = Vector(v["target"])
        cam.dof.focus_distance = (tgt - cam_ob.location).length
        cam.dof.aperture_fstop = dof.get("fstop", 11)


def compositor(cfg):
    post = cfg["views"]["render"].get("post") or {}
    sc = bpy.context.scene
    sc.use_nodes = True
    nt = sc.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    cur = rl.outputs["Image"]
    L_ = nt.links.new
    if post.get("glare"):
        g = nt.nodes.new("CompositorNodeGlare")
        g.glare_type = "FOG_GLOW"
        g.quality = "HIGH"
        g.threshold = 1.0
        g.mix = -1 + 2 * post["glare"]
        L_(cur, g.inputs["Image"])
        cur = g.outputs["Image"]
    if post.get("dispersion"):
        ld = nt.nodes.new("CompositorNodeLensdist")
        ld.inputs["Dispersion"].default_value = post["dispersion"]
        L_(cur, ld.inputs["Image"])
        cur = ld.outputs["Image"]
    if post.get("vignette"):
        em = nt.nodes.new("CompositorNodeEllipseMask")
        em.width, em.height = 0.95, 0.95
        bl = nt.nodes.new("CompositorNodeBlur")
        bl.filter_type = "GAUSS"
        bl.use_relative = True
        bl.factor_x = bl.factor_y = 30
        L_(em.outputs["Mask"], bl.inputs["Image"])
        mr = nt.nodes.new("CompositorNodeMapRange")
        mr.inputs["To Min"].default_value = 1 - post["vignette"]
        mr.inputs["To Max"].default_value = 1.0
        L_(bl.outputs["Image"], mr.inputs["Value"])
        mul = nt.nodes.new("CompositorNodeMixRGB")
        mul.blend_type = "MULTIPLY"
        L_(cur, mul.inputs[1])
        L_(mr.outputs[0], mul.inputs[2])
        cur = mul.outputs["Image"]
    L_(cur, comp.inputs["Image"])


# ---------------------------------------------------------------- interiors
def interior_lights(cfg, on=True):
    col = bpy.data.collections.get("interior_lights")
    if col is None:
        col = bpy.data.collections.new("interior_lights")
        bpy.context.scene.collection.children.link(col)
        for r in cfg.get("rooms", []):
            if r["kind"] != "room":
                continue
            li = bpy.data.lights.new("room", "POINT")
            li.energy = max(25.0, min(140.0, r["area_m2"] * 1.2))
            li.shadow_soft_size = 0.5
            li.color = blackbody(2800)
            ob = bpy.data.objects.new("room_light", li)
            ob.location = (r["x"], r["y"], r["z"])
            col.objects.link(ob)
    for ob in col.objects:
        ob.hide_render = not on
