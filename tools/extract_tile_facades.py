"""Sample facade colours from the client's Google-tiles .3dm (reference only).

For every near-vertical tile triangle: centroid (drawing frame, m, z rel. the
Revit L1), horizontal normal, and the texture colour at its centroid.
Writes cache/facade_samples.npz, used by northstar/site/context.py to colour
and classify rebuilt context facades.

    python tools/extract_tile_facades.py path/to/site.3dm
"""
from __future__ import annotations

import math
import ntpath
import os
import sys
from pathlib import Path

import numpy as np
import rhino3dm as r
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from northstar import spec as specmod  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "cache"


def main(path3dm, radius=200.0):
    sp = specmod.load()
    s = sp["project"]["site"]
    th = math.radians(s["plan_rotation_deg"])
    t = np.array(s["plan_origin_mm"])
    z0 = s["revit_export_level1_mm"]
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    m = r.File3dm.Read(path3dm)
    texdir = CACHE / "tile_textures"
    texdir.mkdir(exist_ok=True)
    for e in m.EmbeddedFiles:
        fn = texdir / ntpath.basename(e.Filename.replace("/", "\\"))
        if not fn.exists():
            e.Write(str(fn))
    cache = {}
    P, N, C = [], [], []
    for o in m.Objects:
        if m.Layers.FindIndex(o.Attributes.LayerIndex).FullPath != "3D SITE":
            continue
        g = o.Geometry
        if not isinstance(g, r.Mesh):
            continue
        mt = m.Materials[o.Attributes.MaterialIndex] if o.Attributes.MaterialIndex >= 0 else None
        tx = mt.GetBitmapTexture() if mt else None
        if tx is None:
            continue
        fn = str(texdir / ntpath.basename(tx.FileName))
        if fn not in cache:
            try:
                cache[fn] = np.asarray(Image.open(fn).convert("RGB"))
            except Exception:
                cache[fn] = None
        T = cache[fn]
        if T is None:
            continue
        nv = len(g.Vertices)
        V = np.array([[g.Vertices[i].X, g.Vertices[i].Y, g.Vertices[i].Z] for i in range(nv)])
        UV = np.array([[g.TextureCoordinates[i].X, g.TextureCoordinates[i].Y] for i in range(nv)])
        F = []
        for i in range(len(g.Faces)):
            a = g.Faces[i]
            F.append((a[0], a[1], a[2]))
            if a[2] != a[3]:
                F.append((a[0], a[2], a[3]))
        F = np.array(F)
        xy = (R.T @ (V[:, :2] - t).T).T / 1000.0
        z = (V[:, 2] - z0) / 1000.0
        W = np.c_[xy, z]
        a, b, c = W[F[:, 0]], W[F[:, 1]], W[F[:, 2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1) + 1e-12
        n = n / ln[:, None]
        cen = (a + b + c) / 3
        keep = (np.abs(n[:, 2]) < 0.35) & (np.hypot(cen[:, 0], cen[:, 1]) < radius) & (ln > 1e-3)
        if not keep.any():
            continue
        uv = (UV[F[keep, 0]] + UV[F[keep, 1]] + UV[F[keep, 2]]) / 3
        tu = np.clip((uv[:, 0] * T.shape[1]).astype(int), 0, T.shape[1] - 1)
        tv = np.clip(((1 - uv[:, 1]) * T.shape[0]).astype(int), 0, T.shape[0] - 1)
        P.append(cen[keep])
        hn = n[keep][:, :2]
        hn /= np.linalg.norm(hn, axis=1, keepdims=True) + 1e-12
        N.append(hn)
        C.append(T[tv, tu])
    P, N, C = np.concatenate(P), np.concatenate(N), np.concatenate(C)
    np.savez_compressed(CACHE / "facade_samples.npz", P=P.astype(np.float32), N=N.astype(np.float32),
                        C=C.astype(np.uint8), z_ref_mm=z0)
    print(len(P), "facade samples")


if __name__ == "__main__":
    main(sys.argv[1])
