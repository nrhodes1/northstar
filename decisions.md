# Decisions log

Every modelling assumption that is not dimensioned in the drawings, or that
departs from them, is recorded here with its source. Open items are marked
**OPEN** and need an answer from the design team.

| ID | Decision | Source / rationale | Status |
|----|----------|--------------------|--------|
| D-001 | Drawing frame: origin at grid 1/A, +X toward grid 9, +Y toward grid A (NE). Levels relative to L1 = 0 (drawings label L1 as 10'-0"). | A1.01–A1.04 | firm |
| D-002 | Grids 1–9 at 11'-6"; A–B 12'-4"; B–C 12'-10" (A–C 25'-2"). | Measured from the vector PDF and calibrated on grids 1–9 (scale error < 0.1%) | firm |
| D-003 | Exterior outlines per storey traced from the PDF wall linework (outside face of the drawn 16" wall), cleaned by hand. L1 sits back 1'-0" from the upper floors on the NE, NW and SE faces. | tools/extract_pdf.py, A1.01–A1.04 | firm |
| D-004 | Georeferencing: drawing frame → client site model is a rotation of −47.174° plus a translation of (−3898.8, 13772.2) mm. | 16 porch columns in the SD plans matched to the Revit export in the site .3dm, residual ≤ 4 mm | firm |
| D-005 | Opening locations and sizes (W1 2'-8"×6'-8", W2 5'-4"×6'-8", W3 2'-0"×7'-0", D1 3'-0"×7'-0", D2 6'-0"×7'-0" slider, SF storefront) come from the Revit families in the site .3dm. The SD set doesn't schedule them. | tools/seed_openings_from_revit.py | review |
| D-006 | Structure: load-bearing prefabricated straw panels (Ecococon basis of design, 400 mm) with CLT floors (5-ply, 175 mm) in platform construction. Interior glulam lines on posts where spans exceed 18'. | Client direction 2026-11-21 | firm |
| D-007 | Wall assembly, outside to inside: corrugated metal 7/8", ventilated cavity 1 1/2", wood-fibre sheathing 5/8", straw panel 400 mm, clay plaster 3/4" = 19 1/2". The exterior face is held at the drawn outline, so the interior shrinks about 3 1/2" compared with the drawn 16" walls. | Client direction (corrugated metal for now); Ecococon literature | review |
| D-008 | Windows sit to the interior: the frame's inner face is flush with the inner face of the straw panel. Exterior reveals are lined with wood, and the sills get a sloped metal flashing. | Client direction; redline on G0.00 ("bring window inset for shading") | firm |
| D-009 | Floor build-up: finish 3/4" + gypcrete 1 1/2" + acoustic mat 1/2" + CLT 175 mm = 9 5/8". The 10'-0" floor-to-floor gives about 8'-11" ceilings with exposed CLT. | Assumption; see the A3.00 redline on floor-to-floor | **OPEN** |
| D-010 | Bearing lines: grid B (full length), grid C inside both wings, and y = −36' in the NW wedge. Posts at ≤ 11'-6" spacing, snapped into partitions that stack where possible. The build report lists posts that land in rooms. | Structural assumption pending an engineer | **OPEN** |
| D-011 | Gallery: HSS 6" columns at the drawn locations, steel-framed decks with wood decking, 42" guards with 3/4" pickets at 3 1/2" clear, U-stairs inside the drawn stair footprints, PV canopy at L4. All materials are swappable in gallery.yaml. | A1.01–A1.04, rendering; client direction | review |
| D-012 | The NW stair footprint (10'-8 1/2" deep) can't fit two 9-riser flights plus a 3'-6" landing with 11" treads. The build check fails at 10 13/16" treads. | IBC 1011.5.2 | **OPEN** — lengthen the footprint about 2" or use a different configuration |
| D-013 | Balconies: the drawn recess plus a 1'-0" projection (NW and SE notches), or a 3'-0" recess plus 2'-6" to 3'-0" projection (NE face). Decks are steel-framed with wood decking, sharing the gallery guard. | A1.01–A1.03 labels and areas | review |
| D-014 | Roof: green roof (sedum) over CLT + 8" wood-fibre insulation; PV in a 45° herringbone with 1'-6" setbacks, as drawn on A1.05. IFC 1205 access pathways need verification. | A1.05 | **OPEN** (fire access) |
| D-015 | Parapet 2'-4" above top of roof CLT with metal coping. Terrace curbs 8"; guards as at the gallery. | A2.00 (about 2'-4" measured) | review |
| D-016 | L1 FFE in the site model is 13.614 m (Revit), but USGS lidar (2021) puts existing grade 0.5–1.1 m below that. The model keeps the Revit FFE for now and grades a pad around the building; see site.yaml `ffe_override_m`. | USGS 3DEP 1 m DEM vs the site model | **OPEN** — needs a survey |
| D-017 | Google tiles are used only as reference (orthophoto, building heights, tree canopy). All context is rebuilt from City of Boston GIS: building footprints with roof heights, sidewalks, street centrelines, parcels, street trees. | Client direction | firm |
