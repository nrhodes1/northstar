"""Download the Poly Haven assets listed in spec/assets.yaml into cache/assets/.

    python tools/fetch_assets.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "cache" / "assets"
API = "https://api.polyhaven.com/files/"
UA = {"User-Agent": "northstar-pipeline"}


def get(url, path: Path):
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def files(asset):
    req = urllib.request.Request(API + asset, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def jobs():
    spec = yaml.safe_load(open(ROOT / "spec" / "assets.yaml"))
    res = spec.get("resolution", "2k")
    out = []
    h = spec["hdri"]
    f = files(h["id"])
    out.append((f["hdri"][h.get("res", "4k")]["exr"]["url"], DEST / "hdri" / f"{h['id']}.exr"))
    for t in spec["textures"]:
        f = files(t)
        for key, name in (("Diffuse", "diff"), ("nor_gl", "nor"), ("Rough", "rough")):
            if key in f:
                out.append((f[key][res]["jpg"]["url"], DEST / "textures" / t / f"{name}.jpg"))
    for group, ids in spec["models"].items():
        for m in ids:
            f = files(m)
            b = f["blend"][res]["blend"]
            d = DEST / "models" / m
            out.append((b["url"], d / f"{m}.blend"))
            for rel, inc in (b.get("include") or {}).items():
                out.append((inc["url"], d / rel))
    return out


if __name__ == "__main__":
    js = jobs()
    print(len(js), "files")
    with ThreadPoolExecutor(6) as ex:
        for p in ex.map(lambda j: get(*j), js):
            pass
    tot = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
    print(f"done: {tot / 1e6:.0f} MB in {DEST}")
