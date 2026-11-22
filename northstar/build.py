"""Build the building model from spec/ and export it.

    python -m northstar.build                 # all systems -> out/
    python -m northstar.build --only envelope,glazing
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import spec as specmod
from .builders import envelope, floors, gallery, structure
from .builders.context import Ctx
from .model import Model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

STEPS = [
    ("envelope", lambda c, m, o: (envelope.build_walls(c, m, o), envelope.build_base_flashing(c, m))),
    ("glazing", lambda c, m, o: envelope.build_openings(c, m, o)),
    ("structure", lambda c, m, o: (floors.build_floors(c, m), structure.build_bearing_lines(c, m))),
    ("interiors", lambda c, m, o: structure.build_interiors(c, m)),
    ("roof", lambda c, m, o: floors.build_roofs(c, m)),
    ("gallery", lambda c, m, o: gallery.build_gallery(c, m)),
    ("balconies", lambda c, m, o: gallery.build_balconies(c, m)),
]


def build(only=None, spec_dir=None) -> tuple[Model, Ctx]:
    sp = specmod.load(spec_dir) if spec_dir else specmod.load()
    ctx = Ctx.build(sp)
    model = Model("building")
    openings = envelope.opening_list(ctx)
    for o in openings:
        if o["dist"] > 0.5:
            model.flag(f"Opening {o['id']} is {o['dist'] / 0.3048:.1f}' from nearest wall — check openings.yaml")
    for name, fn in STEPS:
        if only and name not in only:
            continue
        t = time.time()
        fn(ctx, model, openings)
        print(f"  {name:10s} {time.time() - t:5.1f}s")
    return model, ctx


def write_render_config(sp, out):
    """Blender's Python has no PyYAML: hand it resolved JSON."""
    import json
    from .units import L
    mats = {}
    for k, v in sp["materials"].items():
        d = dict(v)
        for key in ("pitch", "depth", "size"):
            if key in d:
                d[key] = L(d[key])
        mats[k] = d
    views = json.loads(json.dumps(sp["views"]))
    for v in views["views"].values():
        v["position"] = [L(c) for c in v["position"]]
        v["target"] = [L(c) for c in v["target"]]
        if "ortho_scale" in v:
            v["ortho_scale"] = L(v["ortho_scale"])
        if "eye_height" in v:
            v["eye_height"] = L(v["eye_height"])
    # cladding rib profiles live in envelope.yaml; attach them to their materials
    for ct in sp["envelope"]["cladding_types"].values():
        prof = ct.get("profile") or {}
        if ct["material"] in mats and prof:
            mats[ct["material"]]["rib"] = {"shape": prof.get("shape", "sine"), "pitch": L(prof["pitch"]),
                                           "depth": L(prof["depth"]),
                                           "orientation": prof.get("orientation", "vertical")}
    site = sp["project"]["site"]
    cfg = {"materials": mats, "views": views,
           "site": {"lat": site["anchor_lat"], "lon": site["anchor_lon"],
                    "plan_rotation_deg": site["plan_rotation_deg"]}}
    with open(out / "render_config.json", "w") as f:
        json.dump(cfg, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--frame", default="site", choices=["site", "plan"])
    a = ap.parse_args(argv)
    only = set(x for x in a.only.split(",") if x)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, ctx = build(only)
    from .export import gltf, rhino
    from . import report
    mats = ctx.spec["materials"]
    rhino.write(model, ctx.spec, out / "northstar_building.3dm", frame=a.frame, materials=mats)
    gltf.write(model, out / "northstar_building.glb", materials=mats)
    report.write(model, ctx, out / "build_report.md")
    write_render_config(ctx.spec, out)
    print(f"{len(model.elements)} elements -> {out}")
    return model


if __name__ == "__main__":
    main()
