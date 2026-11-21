#! python3
# Rhino 8 (Script Editor, CPython 3) — rebuild + (re)load North Star into the
# open document. Works on its own or via the Rhino MCP tool
# `execute_rhinoscript_python_code` (Claude runs it for you).
#
# 1. Runs the spec build (python -m northstar.build) with the repo's venv /
#    system Python, which writes out/northstar_building.3dm in the client
#    site-model frame (mm, georeferenced to the Google-tiles .3dm).
# 2. Deletes previously loaded North Star layers and imports the new file
#    (and the context, if requested), so the open site model stays intact.
import os
import subprocess
import sys

import Rhino
import scriptcontext as sc

REPO = os.environ.get("NORTHSTAR_REPO", os.path.join(os.path.expanduser("~"), "northstar"))
PYTHON = os.environ.get("NORTHSTAR_PYTHON", "python")
LOAD_SITE = False            # set True to also load the rebuilt context
ROOT_LAYERS = ["envelope", "glazing", "structure", "interiors", "roof", "gallery", "balconies",
               "landscape", "site", "context"]


def run_build():
    r = subprocess.run([PYTHON, "-m", "northstar.build"], cwd=REPO, capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode:
        print(r.stderr[-4000:])
        raise RuntimeError("build failed")


def purge(doc, roots):
    for name in roots:
        idx = doc.Layers.FindByFullPath(name, -1)
        if idx < 0:
            continue
        layer = doc.Layers[idx]
        for obj in doc.Objects.FindByLayer(layer) or []:
            doc.Objects.Delete(obj, True)
        for child in layer.GetChildren() or []:
            purge(doc, [child.FullPath])
            doc.Layers.Purge(child.Index, True)
        doc.Layers.Purge(idx, True)


def main():
    doc = sc.doc
    run_build()
    roots = ROOT_LAYERS if LOAD_SITE else ROOT_LAYERS[:-2]
    purge(doc, roots)
    doc.Import(os.path.join(REPO, "out", "northstar_building.3dm"))
    if LOAD_SITE:
        doc.Import(os.path.join(REPO, "out", "northstar_site.3dm"))
    doc.Views.Redraw()
    print("North Star reloaded")


main()
