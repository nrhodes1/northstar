# North Star House: spec-driven building model

466 River Street, Mattapan, Boston (Habitat for Humanity / MASS Design Group).

Every part of the building is described in **`spec/*.yaml`**. Python builders turn the spec
into geometry, which is exported to **Rhino** (`.3dm`, layered and georeferenced onto the
Google-tiles site model) and **Blender** (`.glb`, then a photoreal scene with the rebuilt
site context). The spec is the only place design decisions live.

```
spec/*.yaml ──► northstar/builders ──► out/northstar_building.3dm ──► Rhino (inspect, mark up)
     │                              └─► out/northstar_building.glb ─┐
     ├──► northstar/site (Boston GIS + USGS lidar, tiles as reference) ──► out/site.glb ─┤
     ├──► checks/ (IBC / zoning / constructability) ──► out/build_report.md             │
     └──► spec/views.yaml + materials.yaml ──► blender/build_scene.py ◄─────────────────┘
                                                   └─► out/renders/*.png, out/northstar.blend
```

## Where to change things

| You want to change… | Edit |
|---|---|
| Storey outlines / massing | `spec/massing.yaml` |
| Cladding: one line, `cladding:` in `spec/envelope.yaml` (corrugated_metal / standing_seam / wood_rainscreen); add new types under `cladding_types` | `spec/envelope.yaml` |
| Window types, how deep windows sit in the wall | `spec/glazing.yaml` (`placement.position`, `types`) |
| Individual window/door positions | `spec/openings.yaml` |
| Floor/roof build-ups, CLT, bearing lines, posts | `spec/structure.yaml` |
| Gallery, stairs, guardrails, canopy, and all of their materials | `spec/gallery.yaml` |
| Balconies | `spec/balconies.yaml` |
| Green roof, PV layout, terraces | `spec/roofs.yaml` |
| Interior partitions | `spec/interiors/L*.yaml` |
| Colours / shaders of any material | `spec/materials.yaml` |
| Cameras, sun date/time, render settings | `spec/views.yaml` |
| Code and zoning limits used by the checks | `spec/code.yaml` |
| Site extent, grading, surfaces, our landscape | `spec/site.yaml` |
| How context buildings and trees are rebuilt | `spec/context.yaml` |

Assumptions and open questions: **`decisions.md`**. Latest checks and quantities:
`out/build_report.md` (regenerated on every build; a copy is in `docs/`).

## Run

```bash
pip install -r requirements.txt
python -m northstar.build                 # building -> out/
python -m northstar.site.build            # site + context -> out/ (needs cache/, see below)
python -m northstar.drawings              # annotated review plans -> out/drawings/plans.pdf
blender -b -P blender/build_scene.py -- --out out --view all --samples 128 --save out/northstar.blend
python -m pytest -q
```

Useful flags: `--view porch_street,aerial_sw`, `--samples 32 --res 1280x720` for drafts,
and `--time 09:30` or `--date 2027-06-21` to restudy the light.

### Site data cache (`cache/`, not committed)
`spec/site.yaml → sources` lists every dataset: USGS 3DEP 1 m DEM, and the Boston building
footprints with roof heights, sidewalk inventory, street segments, 2024 parcels and BPRD
street trees. The client's Google-tiles `.3dm` (on Google Drive) is turned into
`cache/tiles_plan.npz` and an orthophoto `cache/ortho_filled.npz`. Those are used only as
reference: ground cover, roof forms and heights, buildings built after 2010, and tree
canopy. See `tools/` for the extractors.

## Rhino and Blender on your PC (Windows)

1. Clone this repo to `C:\Users\<you>\northstar` and run `pip install -r requirements.txt`.
2. **Rhino 8.20+**: install `rhinomcp` from the Package Manager and run `mcpstart`. Then
   `claude mcp add rhino -- uvx rhinomcp@latest`.
   `rhino/load_in_rhino.py` rebuilds the model and swaps it into the open site model, keeping
   the Google tiles. Claude can run it through the MCP.
3. **Blender 4.2**: `uvx mcp-for-blender install-addon`, enable it, then
   `claude mcp add blender uvx mcp-for-blender`. Open `out/northstar.blend`, or rebuild it
   with the command above. GPU renders are much faster than the cloud CPU drafts.
4. Run `claude remote-control` in the repo folder to drive it from your phone.

## Frames and units
- Spec numbers are **feet**. Strings may carry units (`"11'-6\""`, `"3/4\""`, `"400mm"`).
  Internally everything is metres.
- Drawing frame: origin at grid 1/A, +X toward grid 9, +Y toward grid A (NE). L1 = 0.
- Site frame (Rhino exports): the client's Google-tiles model, in mm, with Z = NAVD88. The
  transform is in `spec/project.yaml → site`.
