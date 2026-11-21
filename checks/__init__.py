"""Design checks run on every build. Each returns {name, status, detail}.

Add a check: write a function `check_xxx(model, ctx) -> dict | list[dict]`
and list it in CHECKS. Constraint values come from spec/code.yaml.
"""
from __future__ import annotations

import math

from northstar.units import FT, L, fmt_ftin

SF = FT * FT


def _r(name, ok, detail, warn=False):
    return {"name": name, "status": "pass" if ok else ("warn" if warn else "fail"), "detail": detail}


def check_guards(model, ctx):
    code = ctx.spec["code"]["guards"]
    g = ctx.spec["gallery"]["guardrail"]
    h = L(g["height"])
    gap = L(g["picket"]["spacing"])
    return [
        _r("Guard height (IBC 1015.3)", h >= L(code["min_height"]) - 1e-4,
           f"{fmt_ftin(h)} vs min {code['min_height']}"),
        _r("Guard openings (IBC 1015.4)", gap < L(code["max_opening"]),
           f"clear {fmt_ftin(gap)} vs < {code['max_opening']}"),
    ]


def check_stairs(model, ctx):
    sp = ctx.spec
    code = sp["code"]["stairs"]
    sd = sp["gallery"]["stair_detail"]
    out = []
    import numpy as np
    from northstar.units import P
    for s in sp["gallery"]["stairs"]:
        q = [np.array(P(p)) for p in s["footprint"]]
        W = np.linalg.norm(q[1] - q[0])
        R = np.linalg.norm(((q[3] - q[0]) + (q[2] - q[1])) / 2)
        sw = min(L(sd["width"]), W / 2)
        H = L(10)
        n = math.ceil(H / L(sd["riser_max"]))
        n += n % 2
        per = n // 2
        tread = (R - sw) / (per - 1)
        out.append(_r(f"Stair {s['name']} riser (IBC 1011.5.2)", H / n <= L(code["max_riser"]) + 1e-4,
                      f"{n} risers @ {fmt_ftin(H / n)}"))
        out.append(_r(f"Stair {s['name']} tread (IBC 1011.5.2)", tread >= L(code["min_tread"]) - 1e-3,
                      f"{per - 1} treads @ {fmt_ftin(tread)} in a {fmt_ftin(R)} footprint"))
        out.append(_r(f"Stair {s['name']} width (IBC 1011.2)", sw >= L(code["min_width"]) - 1e-4,
                      f"{fmt_ftin(sw)} (44\" required if occupant load ≥ 50)", warn=True))
    return out


def check_height(model, ctx):
    sp = ctx.spec
    top = max(e.mesh.bounds[1][2] for e in model.elements if e.system in ("envelope", "roof"))
    zmax = L(sp["code"]["zoning"]["max_height"])
    bmax = L(sp["code"]["building"]["max_height"])
    roof = sp.level_z("ROOF")
    return [
        _r("Stories (IBC 504.4)", len(ctx.outlines) <= sp["code"]["building"]["max_stories"],
           f"{len(ctx.outlines)} stories"),
        _r("Building height to roof (IBC 504.3)", roof <= bmax, f"{fmt_ftin(roof)} vs {fmt_ftin(bmax)}"),
        _r("Zoning height to roof (verify district)", roof <= zmax,
           f"{fmt_ftin(roof)} to roof, {fmt_ftin(top)} to top of PV/parapet vs {fmt_ftin(zmax)}", warn=True),
    ]


def check_wwr(model, ctx):
    glass = sum(e.mesh.area / 2 for e in model.elements if e.system == "glazing" and e.layer == "glass")
    wall = 0.0
    for st, g in ctx.outlines.items():
        z0, z1, _ = ctx.storey_span(st)
        wall += g.length * (z1 - z0)
    wwr = glass / wall if wall else 0
    lim = ctx.spec["code"]["energy"]["max_window_wall_ratio"]
    return _r("Window-to-wall ratio (energy)", wwr <= lim,
              f"{wwr:.1%} (glass {glass / SF:,.0f} SF / wall {wall / SF:,.0f} SF) vs {lim:.0%}", warn=True)


def check_posts(model, ctx):
    posts = [e for e in model.elements if e.layer == "posts"]
    bad = [e for e in posts if not e.props.get("in_wall")]
    return _r("Interior posts concealed in stacked walls", not bad,
              f"{len(posts) - len(bad)}/{len(posts)} posts inside partitions", warn=True)


def check_clt_span(model, ctx):
    """Sample each storey's floor plate on a 1' grid; the CLT span at a
    point is the distance between the supports (bearing walls or glulam
    lines) above and below it in the span direction (Y)."""
    import numpy as np
    from shapely.geometry import LineString
    from northstar.builders.structure import bearing_segments
    sp = ctx.spec
    mx = L(sp["structure"]["interior_support"]["clt_max_span"])
    segs = bearing_segments(ctx)
    worst, where = 0.0, None
    for st, g in ctx.outlines.items():
        inner = g.buffer(-ctx.structure_inner, join_style=2)
        minx, miny, maxx, maxy = inner.bounds
        lines = [(y, a, b) for s_, y, a, b in segs if s_ == st]
        for x in np.arange(minx + 0.15, maxx, 0.3):
            cut = LineString([(x, miny - 1), (x, maxy + 1)]).intersection(inner)
            for piece in getattr(cut, "geoms", [cut]):
                if piece.is_empty:
                    continue
                ys = sorted(c[1] for c in piece.coords)
                sup = [ys[0], ys[-1]] + [y for y, a, b in lines if a <= x <= b and ys[0] < y < ys[-1]]
                sup.sort()
                span = max(b - a for a, b in zip(sup[:-1], sup[1:]))
                if span > worst:
                    worst, where = span, (st, x)
    st, x = where
    return _r("CLT span between supports", worst <= mx + 1e-3,
              f"worst {fmt_ftin(worst)} at {st}, x={x / FT:.1f}' (limit {fmt_ftin(mx)})", warn=True)


def check_openings_hosted(model, ctx):
    from northstar.builders.envelope import opening_list
    ol = opening_list(ctx)
    far = [o for o in ol if o["dist"] > 0.5]
    return _r("Openings sit in exterior walls", not far, f"{len(ol) - len(far)}/{len(ol)} hosted")


CHECKS = [check_guards, check_stairs, check_height, check_wwr, check_posts, check_clt_span,
          check_openings_hosted]


def run_all(model, ctx):
    out = []
    for c in CHECKS:
        try:
            r = c(model, ctx)
        except Exception as e:  # keep the build going
            r = _r(c.__name__, False, f"check error: {e}")
        out += r if isinstance(r, list) else [r]
    return out
