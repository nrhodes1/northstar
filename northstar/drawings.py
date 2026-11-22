"""Annotated floor plans generated from the model + design-review analysis.

    python -m northstar.drawings      -> out/drawings/plans.pdf + plan_L*.png

Every plan is cut at 4'-0" above its level. Concerns from analysis.py are
drawn as located markers; "code" items get numbered callouts listed in the
side panel.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon as MPoly  # noqa: E402
from shapely.geometry import LineString, Point  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from . import analysis as A  # noqa: E402
from . import geom as G  # noqa: E402
from .units import FT, L, P, fmt_ftin  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out" / "drawings"
SHEET = (17.0, 11.0)          # tabloid landscape, inches
SCALE = 3 / 32 / 12           # 3/32" = 1'-0"  (paper inch per model inch)
CUT = L("4'-0\"")
SEV = {"code": "#d62728", "coord": "#ff7f0e", "info": "#1f77b4"}
MARK = {"fire_separation": "s", "stair": "D", "post": "X", "wet_stack": "o", "corner": "^", "pier": "v"}
KIND_LABEL = {
    "fire_separation": "Fire separation distance (IBC 705)",
    "stair": "Stair geometry (IBC 1011)",
    "post": "Bearing-line post falls in a room",
    "wet_stack": "Plumbing fixture not stacked",
    "corner": "Non-orthogonal straw-panel corner",
    "pier": "Narrow panel pier at corner/opening",
}


def ft(xy):
    return np.asarray(xy) / FT


def _fill(ax, geom, **kw):
    for p in G.as_polys(geom):
        ax.add_patch(MPoly(ft(np.asarray(p.exterior.coords)), closed=True, **kw))
        for h in p.interiors:
            ax.add_patch(MPoly(ft(np.asarray(h.coords)), closed=True, fc="white", ec=kw.get("ec", "none"),
                               lw=kw.get("lw", 0)))


def _line(ax, coords, **kw):
    c = ft(coords)
    ax.plot(c[:, 0], c[:, 1], **kw)


def opening_rects(ctx, openings, st, z_cut):
    rects = []
    for o in openings:
        if o["storey"] != st:
            continue
        z0 = ctx.spec.level_z(st)
        if not (z0 + o["sill"] <= z0 + z_cut <= z0 + o["head"]):
            continue
        d = G.unit(o["b"] - o["a"])
        n = np.array([d[1], -d[0]])
        a = o["a"]
        u0, u1 = o["u"] - o["w"] / 2, o["u"] + o["w"] / 2
        c = [a + d * u0 + n * 0.05, a + d * u1 + n * 0.05,
             a + d * u1 - n * (ctx.wall_T + 0.05), a + d * u0 - n * (ctx.wall_T + 0.05)]
        rects.append((o, G.polygon(c), a, d, n, u0, u1))
    return rects


def draw_plan(ax, model, ctx, openings, concerns, st):
    sp = ctx.spec
    P_ = ctx.outlines[st]
    rects = opening_rects(ctx, openings, st, CUT)
    holes = unary_union([r[1] for r in rects]) if rects else None

    # site context: lot line, gallery/balconies below/above
    lot = sp["site"].get("lot")
    if lot:
        lp = G.polygon([P(p) for p in lot["points"]])
        _line(ax, np.asarray(lp.exterior.coords), color="#555", lw=0.8, ls=(0, (8, 3, 2, 3)))
    # floor below / above outlines (dashed) for context
    names = list(ctx.outlines)
    i = names.index(st)
    if i + 1 < len(names):
        for p in G.as_polys(ctx.outlines[names[i + 1]]):
            _line(ax, np.asarray(p.exterior.coords), color="#999", lw=0.6, ls="--")

    # walls: layered poche
    colors = {"cladding": "#4a5560", "sheathing": "#8b6b4a", "straw": "#e8d28a", "plaster": "#d9d2c5"}
    for name, d0, d1, mat, s in ctx.layers:
        if mat is None:
            continue
        band = G.ring(P_, d0, d1)
        if holes is not None:
            band = band.difference(holes)
        _fill(ax, band, fc=colors.get(name, "#999"), ec="none", lw=0,
              hatch="...." if name == "straw" else None)
    for p in G.as_polys(P_):
        _line(ax, np.asarray(p.exterior.coords), color="black", lw=1.2)
        _line(ax, np.asarray(G.inset(p, ctx.wall_T).exterior.coords), color="black", lw=0.8)

    # openings
    for o, poly, a, d, n, u0, u1 in rects:
        kind = o["ty"].get("kind", "window")
        f_in = ctx.wall_T - (ctx.wall_T - ctx.structure_inner)
        if kind == "door":
            hinge = a + d * u0 - n * f_in
            r = o["w"]
            th = np.linspace(0, math.pi / 2, 20)
            arc = [hinge + (d * math.cos(t) - n * math.sin(t)) * r for t in th]
            _line(ax, np.array(arc), color="black", lw=0.5)
            _line(ax, np.array([hinge, hinge - n * r]), color="black", lw=0.9)
        else:
            for off in (f_in - L("2\""), f_in - L("3\"")):
                _line(ax, np.array([a + d * u0 - n * off, a + d * u1 - n * off]), color="black", lw=0.5)
            if kind == "slider":
                m = (u0 + u1) / 2
                _line(ax, np.array([a + d * m - n * (f_in - L("1\"")), a + d * m - n * (f_in - L("4\""))]),
                      color="black", lw=0.5)

    # interiors
    from .builders.structure import partitions
    g = partitions(ctx, st)
    if g is not None:
        _fill(ax, g.intersection(G.inset(P_, ctx.wall_T)), fc="#bdbdbd", ec="#333", lw=0.3)

    # structure: posts + beams (dashed, above)
    for e in model.elements:
        if e.level != st:
            continue
        if e.layer == "posts":
            b = e.mesh.bounds
            ax.add_patch(MPoly(ft([[b[0][0], b[0][1]], [b[1][0], b[0][1]], [b[1][0], b[1][1]], [b[0][0], b[1][1]]]),
                               closed=True, fc="#7a4f2a", ec="black", lw=0.4))
        if e.layer == "beams":
            b = e.mesh.bounds
            _line(ax, np.array([[b[0][0], (b[0][1] + b[1][1]) / 2], [b[1][0], (b[0][1] + b[1][1]) / 2]]),
                  color="#7a4f2a", lw=0.7, ls=(0, (6, 3)))

    # gallery deck / stairs / columns
    gl = sp["gallery"]
    deck = G.polygon([P(p) for p in gl["deck"]["outline"]])
    if st in gl["levels_served"]:
        _fill(ax, deck, fc="#efe6d8", ec="black", lw=0.6)
    elif st == "L1":
        _fill(ax, deck, fc="#f3f3f3", ec="#777", lw=0.5, ls="--")
    for s in gl["stairs"]:
        q = np.array([P(p) for p in s["footprint"]])
        ax.add_patch(MPoly(ft(q), closed=True, fc="none", ec="black", lw=0.6))
        w = q[1] - q[0]
        r = (q[3] - q[0] + q[2] - q[1]) / 2
        for k in range(1, 9):
            t = k / 9 * 0.68
            for side in (0.0, 0.5):
                p0 = q[0] + r * t + w * side
                _line(ax, np.array([p0, p0 + w * 0.5]), color="#555", lw=0.35)
        _line(ax, np.array([q[0] + w * 0.5, q[0] + w * 0.5 + r * 0.68]), color="black", lw=0.5)
        c = ft(q.mean(axis=0))
        ax.text(c[0], c[1], "UP" if st in ("L1", "L2") else "DN", fontsize=5, ha="center", va="center")
    cs = L(gl["columns"]["size"])
    for p in gl["columns"]["points"]:
        x, y = P(p)
        ax.add_patch(MPoly(ft([[x - cs / 2, y - cs / 2], [x + cs / 2, y - cs / 2], [x + cs / 2, y + cs / 2],
                               [x - cs / 2, y + cs / 2]]), closed=True, fc="black"))

    # balconies on this level
    for b in sp["balconies"]["balconies"]:
        if b["level"] == st:
            bp = G.polygon([P(p) for p in b["outline"]]).difference(G.inset(P_, ctx.wall_T))
            _fill(ax, bp, fc="#efe6d8", ec="black", lw=0.5)
    # terraces (roof of the storey below) at this level
    if i > 0:
        below = ctx.outlines[names[i - 1]]
        ter = below.difference(P_.buffer(0.02, join_style=2))
        for p in G.as_polys(G.inset(ter, ctx.wall_T)):
            if p.area > 1:
                _fill(ax, p, fc="#e6eef2", ec="#333", lw=0.4, hatch="++")

    # fixtures
    for f in (sp.get("fixtures") or {}).get("fixtures", {}).get(st, []):
        x, y = ft(P(f["at"]))
        ax.add_patch(Circle((x, y), 0.9 if f["type"] == "WC" else 0.6, fc="white", ec="#333", lw=0.4))

    # rooms
    for r in (sp.get("rooms") or {}).get("rooms", {}).get(st, []):
        x, y = r["at"]
        ax.text(x, y + 1.4, r["name"], fontsize=5.2, ha="center", va="center")
        ax.text(x, y, r["id"], fontsize=5.2, ha="center", va="center",
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="black", lw=0.4))
        if r.get("area_sf"):
            ax.text(x, y - 1.4, f"{r['area_sf']:,.0f} SF", fontsize=4.6, ha="center", va="center")

    # grids
    gx = {k: sp.grid("x", k) / FT for k in sp["project"]["grids"]["x"]}
    gy = {k: sp.grid("y", k) / FT for k in sp["project"]["grids"]["y"]}
    y_top, y_bot = 16, -62
    for k, x in gx.items():
        ax.plot([x, x], [y_bot + 4, y_top - 4], color="#9a9a9a", lw=0.4, ls=(0, (10, 3, 2, 3)), zorder=0)
        for yy in (y_top, y_bot):
            ax.add_patch(Circle((x, yy), 2.0, fc="white", ec="black", lw=0.5, zorder=5))
            ax.text(x, yy, k, fontsize=6, ha="center", va="center", zorder=6)
    for k, y in gy.items():
        ax.plot([-14, 106], [y, y], color="#9a9a9a", lw=0.4, ls=(0, (10, 3, 2, 3)), zorder=0)
        for xx in (-16, 108):
            ax.add_patch(Circle((xx, y), 2.0, fc="white", ec="black", lw=0.5, zorder=5))
            ax.text(xx, y, k, fontsize=6, ha="center", va="center", zorder=6)
    # dimension string across the top
    xs = sorted(gx.values())
    yd = y_top - 5
    ax.plot([xs[0], xs[-1]], [yd, yd], color="black", lw=0.4)
    for x in xs:
        ax.plot([x, x], [yd - 0.6, yd + 0.6], color="black", lw=0.4)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.text((x0 + x1) / 2, yd + 0.6, fmt_ftin((x1 - x0) * FT).replace(" 0/1", ""), fontsize=4.5, ha="center")

    # concerns
    numbered = []
    for c in concerns:
        if c["storey"] != st:
            continue
        x, y = ft(c["xy"])
        col = SEV[c["severity"]]
        if c["kind"] == "fire_separation" and c["severity"] == "info":
            ax.text(x, y, f"FSD {c['title'].split()[1]}", fontsize=4.2, color=col, ha="center", va="center",
                    rotation=0, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=col, lw=0.3))
            continue
        ax.plot(x, y, MARK[c["kind"]], ms=5 if c["kind"] in ("corner", "pier") else 7, mfc="none" if
                c["severity"] != "code" else col, mec=col, mew=1.0, zorder=8, alpha=0.9)
        if c["severity"] == "code":
            numbered.append(c)
    # number code items
    for k, c in enumerate(numbered, 1):
        x, y = ft(c["xy"])
        ax.annotate(str(k), (x, y), xytext=(x + 5, y + 5), fontsize=6.5, color="white", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.25", fc=SEV["code"], ec="none"),
                    arrowprops=dict(arrowstyle="-", color=SEV["code"], lw=0.7), zorder=9)
    return numbered


def north_arrow(ax, ctx, x, y, size=5):
    th = math.radians(ctx.spec["project"]["site"]["plan_rotation_deg"])
    n = np.array([math.sin(-th) * -1, math.cos(th)])  # true north in drawing frame
    n = np.array([-math.sin(th), math.cos(th)])
    n = n / np.linalg.norm(n)
    tip = np.array([x, y]) + n * size
    ax.annotate("", xy=tip, xytext=(x, y), arrowprops=dict(arrowstyle="-|>", color="black", lw=1))
    ax.text(*(tip + n * 1.8), "N", ha="center", va="center", fontsize=7, weight="bold")


def sheet(model, ctx, openings, concerns, st, loss):
    fig = plt.figure(figsize=SHEET)
    # plan axes sized for exact scale
    plan_w_in, plan_h_in = 12.0, 8.9
    ax = fig.add_axes([0.3 / SHEET[0], 1.5 / SHEET[1], plan_w_in / SHEET[0], plan_h_in / SHEET[1]])
    span_x = plan_w_in / SCALE / 12
    span_y = plan_h_in / SCALE / 12
    cx, cy = 46, -22
    ax.set_xlim(cx - span_x / 2, cx + span_x / 2)
    ax.set_ylim(cy - span_y / 2, cy + span_y / 2)
    ax.set_aspect("equal")
    ax.axis("off")
    numbered = draw_plan(ax, model, ctx, openings, concerns, st)
    north_arrow(ax, ctx, cx + span_x / 2 - 8, cy + span_y / 2 - 12)

    # side panel
    side = fig.add_axes([12.6 / SHEET[0], 0.35 / SHEET[1], 4.1 / SHEET[0], 10.3 / SHEET[1]])
    side.axis("off")
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    yy = 0.98
    side.text(0, yy, "AREAS OF POTENTIAL CONCERN", fontsize=9, weight="bold", va="top")
    yy -= 0.035
    for k, c in enumerate(numbered, 1):
        side.text(0, yy, str(k), fontsize=6.5, color="white", va="top", ha="center",
                  bbox=dict(boxstyle="circle,pad=0.2", fc=SEV["code"], ec="none"))
        import textwrap
        body = textwrap.fill(f"{c['title']}. {c['text']}", 70)
        side.text(0.06, yy, body, fontsize=5.6, va="top")
        yy -= 0.012 * (body.count("\n") + 1) + 0.014
    yy -= 0.01
    side.text(0, yy, "Located markers (this floor)", fontsize=7.5, weight="bold", va="top")
    yy -= 0.03
    from collections import Counter
    cnt = Counter((c["kind"], c["severity"]) for c in concerns if c["storey"] == st)
    for (kind, sev), n in sorted(cnt.items(), key=lambda kv: ["code", "coord", "info"].index(kv[0][1])):
        side.plot(0.02, yy - 0.006, MARK[kind], ms=6, mfc=SEV[sev] if sev == "code" else "none", mec=SEV[sev],
                  mew=1)
        lab = KIND_LABEL[kind] + (" - labels on walls" if kind == "fire_separation" and sev == "info" else "")
        side.text(0.06, yy, f"{lab}  ({n})", fontsize=6, va="top")
        yy -= 0.024
    # one-line explanations of coordination kinds
    expl = {
        "post": "Glulam bearing line at grid B/C needs a post where no partition stacks above/below.",
        "wet_stack": "WC/tub with no wet fixture within 4' on the floor below: horizontal drain in the CLT floor zone.",
        "corner": "Angled corners need a custom straw-panel corner detail (New Frameworks to confirm).",
        "pier": "Opening closer to a corner than one min. panel + wall depth: site-built infill.",
    }
    for kind, txt in expl.items():
        if any(c["kind"] == kind and c["storey"] == st for c in concerns):
            import textwrap
            t = textwrap.fill(f"{KIND_LABEL[kind]}: {txt}", 75)
            side.text(0, yy, t, fontsize=5.3, va="top", color="#444")
            yy -= 0.011 * (t.count("\n") + 1) + 0.008
    yy -= 0.01
    side.text(0, yy, "Envelope", fontsize=7.5, weight="bold", va="top")
    yy -= 0.025
    lay = ", ".join(f"{n} {fmt_ftin(d1 - d0)}" for n, d0, d1, m, s in ctx.layers)
    import textwrap
    t = textwrap.fill(f"Wall {fmt_ftin(ctx.wall_T)} ({lay}). Interior area lost vs the drawn 16\" wall on "
                      f"this floor: ~{dict(loss)[st]:,.0f} SF.", 75)
    side.text(0, yy, t, fontsize=5.6, va="top")
    yy -= 0.011 * (t.count("\n") + 1) + 0.02
    # legend swatches
    for lab, col, hatch in (("Cladding", "#4a5560", None), ("Straw panel 400 mm", "#e8d28a", "...."),
                            ("Clay plaster", "#d9d2c5", None), ("Partition", "#bdbdbd", None),
                            ("Gallery / balcony deck", "#efe6d8", None), ("Terrace", "#e6eef2", "++")):
        side.add_patch(MPoly([[0, yy - 0.012], [0.05, yy - 0.012], [0.05, yy], [0, yy]], closed=True, fc=col,
                             ec="black", lw=0.3, hatch=hatch))
        side.text(0.07, yy - 0.006, lab, fontsize=5.6, va="center")
        yy -= 0.02
    side.plot([0, 0.05], [yy - 0.006] * 2, color="#555", lw=0.8, ls=(0, (8, 3, 2, 3)))
    side.text(0.07, yy - 0.006, "Property line (A1.00)", fontsize=5.6, va="center")
    yy -= 0.02
    side.plot([0, 0.05], [yy - 0.006] * 2, color="#7a4f2a", lw=0.8, ls=(0, (6, 3)))
    side.text(0.07, yy - 0.006, "Glulam beam above / post", fontsize=5.6, va="center")

    # title block
    tb = fig.add_axes([0.3 / SHEET[0], 0.3 / SHEET[1], 12.0 / SHEET[0], 1.0 / SHEET[1]])
    tb.axis("off")
    tb.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0", fc="none", ec="black", lw=0.8,
                                transform=tb.transAxes))
    p = ctx.spec["project"]["project"]
    lv = ctx.spec.level_z(st) / FT + ctx.spec["project"].get("drawing_label_offset_ft", 0)
    tb.text(0.015, 0.68, f"{p['name'].upper()}  |  {p['address']}", fontsize=9, weight="bold", va="center")
    tb.text(0.015, 0.3, f"ANNOTATED FLOOR PLAN - {st} (EL. {lv:.0f}'-0\")   |   cut at 4'-0\"   |   "
                        f"generated from spec/ by northstar.drawings - REVIEW SET, NOT FOR CONSTRUCTION",
            fontsize=6.5, va="center")
    tb.text(0.985, 0.5, f"P-{list(ctx.outlines).index(st) + 1:02d}", fontsize=18, weight="bold", ha="right",
            va="center")
    tb.text(0.86, 0.5, 'SCALE 3/32" = 1\'-0"\n(at 11x17)', fontsize=6.5, ha="center", va="center")
    return fig


def main():
    from .build import build
    from .builders.envelope import opening_list
    model, ctx = build()
    openings = opening_list(ctx)
    concerns = A.all_concerns(model, ctx, openings)
    loss = A.area_loss(ctx)
    OUT.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT / "plans.pdf") as pdf:
        for st in ctx.outlines:
            fig = sheet(model, ctx, openings, concerns, st, loss)
            pdf.savefig(fig)
            fig.savefig(OUT / f"plan_{st}.png", dpi=170)
            plt.close(fig)
    # summary for the report
    import json
    with open(OUT / "concerns.json", "w") as f:
        json.dump([{k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()} for c in concerns], f,
                  indent=1, default=float)
    print(f"{len(concerns)} concerns -> {OUT}")


if __name__ == "__main__":
    main()
