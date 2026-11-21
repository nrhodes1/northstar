"""Build report: quantities, schedules, flags, and check results."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

from .units import FT, fmt_ftin

SF = FT * FT
CF = FT ** 3


def write(model, ctx, path, checks=None):
    from checks import run_all
    results = checks if checks is not None else run_all(model, ctx)
    sp = ctx.spec
    lines = [f"# Build report — {sp['project']['project']['name']}",
             f"_Generated {datetime.now():%Y-%m-%d %H:%M} from spec/_", ""]
    lines += ["## Checks", "", "| | Check | Result |", "|---|---|---|"]
    for r in results:
        icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[r["status"]]
        lines.append(f"| {icon} | {r['name']} | {r['detail']} |")
    lines += ["", "## Envelope", "",
              f"- Wall assembly: {sp['envelope']['default_assembly']} — total thickness "
              f"{fmt_ftin(ctx.wall_T)}",
              "- Layers (out → in): " + ", ".join(f"{n} {fmt_ftin(d1 - d0)}" for n, d0, d1, _, _ in ctx.layers),
              f"- Floor build-up {fmt_ftin(ctx.floor_T)}; roof build-up above CLT {fmt_ftin(ctx.roof_T)}", ""]
    # quantities
    vol = defaultdict(float)
    for e in model.elements:
        try:
            vol[(e.system, e.layer, e.material)] += abs(e.mesh.volume)
        except Exception:
            pass
    lines += ["## Quantities (by element group)", "", "| System | Group | Material | Volume (CF) |", "|---|---|---|---|"]
    for (s, l, m), v in sorted(vol.items()):
        lines.append(f"| {s} | {l} | {m} | {v / CF:,.0f} |")
    # openings schedule
    cnt = Counter()
    for e in model.elements:
        if e.system == "glazing" and e.layer == "frames":
            cnt[(e.props.get("type"), e.props.get("storey"))] += 1
    lines += ["", "## Opening schedule", "", "| Type | " + " | ".join(ctx.outlines) + " | Total |",
              "|---|" + "---|" * (len(ctx.outlines) + 1)]
    for t in sorted({k[0] for k in cnt}):
        row = [cnt[(t, s)] for s in ctx.outlines]
        lines.append(f"| {t} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    # areas
    lines += ["", "## Gross floor area (outside face)", ""]
    tot = 0
    for st, g in ctx.outlines.items():
        a = g.area / SF
        tot += a
        lines.append(f"- {st}: {a:,.0f} SF")
    lines.append(f"- **Total: {tot:,.0f} SF**")
    lines += ["", "## Build log / flags", ""] + [f"- {n}" for n in model.notes]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return results
