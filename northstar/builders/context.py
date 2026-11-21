"""Shared build context: resolved spec values used by several builders."""
from __future__ import annotations

from dataclasses import dataclass, field

from shapely.ops import unary_union

from ..geom import polygon
from ..spec import Spec
from ..units import L, P


@dataclass
class Ctx:
    spec: Spec
    outlines: dict = field(default_factory=dict)      # storey -> shapely geometry (m)
    parts: dict = field(default_factory=dict)         # storey -> [(name, Polygon)]
    layers: list = field(default_factory=list)        # [(name, d0, d1, material, structural)]
    wall_T: float = 0.0
    floor_T: float = 0.0
    roof_T: float = 0.0                                # build-up above roof CLT
    roof_clt: float = 0.0

    @classmethod
    def build(cls, spec: Spec) -> "Ctx":
        c = cls(spec)
        for st, plist in spec["massing"]["storeys"].items():
            ps = [(p["name"], polygon([P(xy) for xy in p["points"]])) for p in plist]
            c.parts[st] = ps
            c.outlines[st] = unary_union([p for _, p in ps])
        env = spec["envelope"]
        asm = env["assemblies"][env["default_assembly"]]
        clad = env["cladding_types"][asm["cladding"]]
        d = 0.0
        for lay in asm["layers"]:
            t = L(lay["thickness"])
            mat = lay.get("material")
            if mat == "CLADDING":
                mat = clad["material"]
            c.layers.append((lay["name"], d, d + t, mat, bool(lay.get("structural"))))
            d += t
        c.wall_T = d
        st = spec["structure"]
        c.floor_T = sum(L(x["thickness"]) for x in st["floors"]["assembly"])
        ra = st["roof"]["assembly"]
        c.roof_clt = sum(L(x["thickness"]) for x in ra if x.get("structural"))
        c.roof_T = sum(L(x["thickness"]) for x in ra if not x.get("structural"))
        return c

    # convenience
    def layer(self, name):
        for x in self.layers:
            if x[0] == name:
                return x
        raise KeyError(name)

    @property
    def structure_face(self) -> float:
        """Depth (from outline) of the outer face of the structural layer."""
        for name, d0, d1, mat, s in self.layers:
            if s:
                return d0
        return 0.0

    @property
    def structure_inner(self) -> float:
        for name, d0, d1, mat, s in self.layers:
            if s:
                return d1
        return self.wall_T

    def storey_span(self, st):
        z0 = self.spec.level_z(st)
        above = self.spec.level_above(st)
        z1 = self.spec.level_z(above) if above else z0 + L(10)
        return z0, z1, above

    def storey_above(self, st):
        a = self.spec.level_above(st)
        return a if a in self.outlines else None

    def storey_below(self, st):
        names = list(self.outlines)
        i = names.index(st)
        return names[i - 1] if i > 0 else None
