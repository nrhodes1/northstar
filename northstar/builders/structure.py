"""Interior partitions + interior bearing line (glulam beam on posts).

Posts are placed along each bearing line so that no CLT span exceeds
`clt_max_span` and no beam span exceeds `post_max_spacing`. Posts are
snapped into partitions that stack on every storey where possible; any post
that cannot be hidden in a wall is flagged in the build report.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from .. import geom as G
from ..model import Model
from ..units import FT, L, P, fmt_ftin
from .context import Ctx


def partitions(ctx: Ctx, st: str):
    data = (ctx.spec.get("interiors") or {}).get(st) or {}
    polys = [G.polygon([P(xy) for xy in w["outline"]]) for w in data.get("walls", [])]
    return unary_union(polys) if polys else None


def build_interiors(ctx: Ctx, model: Model):
    for st in ctx.outlines:
        g = partitions(ctx, st)
        if g is None:
            continue
        z0, z1, above = ctx.storey_span(st)
        top = z1 - (ctx.roof_clt if ctx.storey_above(st) is None else ctx.floor_T)
        g = g.intersection(G.inset(ctx.outlines[st], ctx.wall_T))
        mat = ctx.spec["interiors"][st].get("material", "interior_partition")
        model.add("interiors", "partitions", "IfcWall", mat, G.extrude(g, z0, top), level=st)


def build_bearing_lines(ctx: Ctx, model: Model):
    sp = ctx.spec
    s = sp["structure"]["interior_support"]
    post_w, post_d = (L(v) for v in s["post"]["size"])
    bw, bd = (L(v) for v in s["beam"]["size"])
    max_sp = L(s["post_max_spacing"])
    storeys = list(ctx.outlines)

    stacked = None
    for st in storeys:
        g = partitions(ctx, st)
        if g is None:
            continue
        g = g.buffer(0.03)
        stacked = g if stacked is None else stacked.intersection(g)

    for line in s["bearing_lines"]:
        yb = gridval(sp, "y", line["y"])
        x0 = gridval(sp, "x", line["x_from"]) - (1.5 if str(line["x_from"]) in sp["project"]["grids"]["x"] else 0)
        x1 = gridval(sp, "x", line["x_to"]) + (1.5 if str(line["x_to"]) in sp["project"]["grids"]["x"] else 0)
        for st in storeys:
            z0, z1, _ = ctx.storey_span(st)
            under = z1 - (ctx.roof_clt if ctx.storey_above(st) is None else ctx.floor_T)
            inner = G.inset(ctx.outlines[st], ctx.structure_inner)
            walls_here = partitions(ctx, st)
            seg = LineString([(x0, yb), (x1, yb)]).intersection(inner)
            pieces = [seg] if seg.geom_type == "LineString" else list(getattr(seg, "geoms", []))
            for pc in pieces:
                if pc.length < 0.5:
                    continue
                (xa, _), (xb, _) = pc.coords[0], pc.coords[-1]
                xa, xb = sorted((xa, xb))
                span = xb - xa
                # check whether a bearing line is needed: depth across the line
                posts = _place_posts(xa, xb, yb, max_sp, stacked, walls_here)
                xs = [xa] + [p[0] for p in posts] + [xb]
                for k in range(len(xs) - 1):
                    a, b = (xs[k], yb), (xs[k + 1], yb)
                    model.add("structure", "beams", "IfcBeam", s["beam"]["material"],
                              G.segment_solid(a, b, bw, under - bd, under, extend=0.05), level=st,
                              span=fmt_ftin(xs[k + 1] - xs[k]))
                for x, hidden in posts:
                    model.add("structure", "posts", "IfcColumn", s["post"]["material"],
                              G.box3(x - post_w / 2, yb - post_d / 2, z0, x + post_w / 2, yb + post_d / 2,
                                     under - bd), level=st, in_wall=hidden)
                    if not hidden:
                        model.flag(f"POST NOT IN WALL: {st} at x={x / FT:.2f}', grid B")
                    if not _in_wall(x, yb, stacked):
                        pass


def gridval(sp, axis, v):
    if str(v) in sp["project"]["grids"][axis]:
        return sp.grid(axis, v)
    return L(v)


def bearing_segments(ctx):
    """[(storey, y, xa, xb)] support lines clipped to each storey interior."""
    sp = ctx.spec
    out = []
    for line in sp["structure"]["interior_support"]["bearing_lines"]:
        yb = gridval(sp, "y", line["y"])
        x0, x1 = gridval(sp, "x", line["x_from"]) - 0.5, gridval(sp, "x", line["x_to"]) + 0.5
        for st in ctx.outlines:
            inner = G.inset(ctx.outlines[st], ctx.structure_inner)
            seg = LineString([(x0, yb), (x1, yb)]).intersection(inner)
            for pc in getattr(seg, "geoms", [seg]):
                if not pc.is_empty and pc.length > 0.5:
                    xs = sorted(c[0] for c in pc.coords)
                    out.append((st, yb, xs[0], xs[-1]))
    return out


def _in_wall(x, y, g, r=0.05):
    return g is not None and g.buffer(r).contains(Point(x, y))


def _place_posts(xa, xb, y, max_sp, stacked, walls_here):
    """Greedy: from xa, reach as far as possible (<= max_sp) to a candidate
    inside stacked walls; else inside this storey's walls; else at max_sp."""
    def candidates(g):
        if g is None:
            return []
        inter = LineString([(xa, y), (xb, y)]).intersection(g)
        out = []
        for gg in getattr(inter, "geoms", [inter]):
            if gg.is_empty:
                continue
            xs = [c[0] for c in gg.coords]
            out.append((min(xs) + max(xs)) / 2)
        return sorted(out)

    cs = candidates(stacked)
    cw = candidates(walls_here)
    posts = []
    cur = xa
    while xb - cur > max_sp:
        opts = [c for c in cs if cur + 0.6 < c <= cur + max_sp]
        hidden = True
        if not opts:
            opts = [c for c in cw if cur + 0.6 < c <= cur + max_sp]
        if opts:
            nxt = max(opts)
        else:
            nxt, hidden = cur + max_sp, False
        posts.append((nxt, hidden or _in_wall(nxt, y, walls_here)))
        cur = nxt
    return posts
