# North Star House — working rules for Claude

This repo is a **spec-driven building model**. The architect controls the
design; Claude edits the spec, the builders, and the pipeline.

## Golden rules
1. **The spec is the source of truth.** Every design decision lives in
   `spec/*.yaml`. Builders (`northstar/builders/`) contain no project-specific
   numbers. If a value is needed, add it to the spec.
2. **Never hand-edit generated files** (`out/`, `.3dm`, `.blend`). Re-run the build.
3. **Log assumptions** in `decisions.md` (D-xxx) whenever something is not
   dimensioned in the drawings or departs from them. Mark open questions **OPEN**.
4. **Locked values:** a spec key with a trailing comment `# LOCKED` must not be
   changed without explicit instruction from the architect.
5. **Checks must pass or be explained.** `checks/` runs on every build; a FAIL
   is either fixed or recorded in decisions.md.
6. Units: spec numbers are feet; strings may carry units ("11'-6\"", "400mm").
   Internally everything is metres.

## Commands
```bash
pip install -r requirements.txt
python -m northstar.build                       # building -> out/*.3dm, *.glb, build_report.md
python -m northstar.site.build                  # terrain + context -> out/site.glb, out/site.3dm
blender -b -P blender/build_scene.py -- --out out --view all --samples 128   # renders
python -m pytest -q
```

## Layout
- `spec/` project, massing, envelope, structure, glazing, openings, gallery,
  balconies, roofs, interiors/, materials, views, code, site, context
- `northstar/builders/` envelope, floors/roofs/PV, structure, gallery
- `northstar/site/` geo transforms, terrain, context buildings, trees
- `northstar/export/` Rhino (.3dm, layered, site-georeferenced) and glTF
- `blender/` scene assembly, materials, lighting, cameras, scatter
- `checks/` code/zoning/constructability checks (values from spec/code.yaml)
- `tools/` one-off extractors used to seed the spec from the drawings
