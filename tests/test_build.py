import math

import pytest

from northstar import spec as specmod
from northstar.builders.context import Ctx
from northstar.units import FT, IN, L, fmt_ftin


def test_units():
    assert L("11'-6\"") == pytest.approx(11.5 * FT)
    assert L("-12'-4\"") == pytest.approx(-(12 + 4 / 12) * FT)
    assert L("1 1/2\"") == pytest.approx(1.5 * IN)
    assert L("400mm") == pytest.approx(0.4)
    assert L(10) == pytest.approx(10 * FT)
    assert fmt_ftin(L("12'-4 1/2\"")) == "12'-4 1/2\""


def test_spec_frame():
    sp = specmod.load()
    assert sp.grid("x", "9") == pytest.approx(92 * FT)
    assert list(sp.levels) == ["L1", "L2", "L3", "L4", "ROOF"]


def test_wall_assembly():
    ctx = Ctx.build(specmod.load())
    names = [l[0] for l in ctx.layers]
    assert names[0] == "cladding" and "straw" in names
    assert ctx.wall_T == pytest.approx(L("19 1/2\"") + (L("400mm") - L("15 3/4\"")), abs=0.002)


@pytest.fixture(scope="module")
def built():
    from northstar.build import build
    return build()


def test_build_produces_systems(built):
    model, ctx = built
    systems = {e.system for e in model.elements}
    assert {"envelope", "glazing", "structure", "roof", "gallery", "balconies"} <= systems


def test_all_openings_hosted(built):
    model, ctx = built
    from northstar.builders.envelope import opening_list
    assert all(o["dist"] < 0.5 for o in opening_list(ctx))


def test_windows_set_to_interior(built):
    """Window frames must sit at the inner face of the straw panel."""
    model, ctx = built
    plaster = ctx.wall_T - ctx.structure_inner
    frames = [e for e in model.elements if e.layer == "frames" and e.props.get("type") == "W1"]
    assert frames
    # frame centroid distance from outline should be ~ wall_T - plaster - frame_depth/2
    e = frames[0]
    poly = ctx.outlines[e.level]
    from shapely.geometry import Point
    c = e.mesh.centroid
    d = poly.exterior.distance(Point(c[0], c[1]))
    assert d == pytest.approx(ctx.wall_T - plaster - L("4 1/2\"") / 2, abs=0.03)


def test_checks_run(built):
    model, ctx = built
    from checks import run_all
    res = run_all(model, ctx)
    assert res and all(r["status"] in ("pass", "warn", "fail") for r in res)
    guard = [r for r in res if r["name"].startswith("Guard height")][0]
    assert guard["status"] == "pass"
