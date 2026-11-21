"""Model container: every generated element with semantics."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import trimesh


@dataclass
class Element:
    id: str
    system: str          # e.g. envelope, glazing, structure, gallery, roof, site
    layer: str           # Rhino layer path below the system, e.g. "walls/cladding"
    ifc: str             # IFC class, e.g. IfcWall, IfcWindow, IfcSlab
    material: str        # key into materials.yaml
    mesh: trimesh.Trimesh
    level: str | None = None
    props: dict = field(default_factory=dict)

    @property
    def path(self) -> str:
        return f"{self.system}::{self.layer}"


class Model:
    def __init__(self, name="model"):
        self.name = name
        self.elements: list[Element] = []
        self.curves: list[dict] = []     # annotation polylines (grids, panel joints)
        self.notes: list[str] = []       # build log / flags
        self._ids = Counter()

    def add(self, system, layer, ifc, material, mesh, level=None, id=None, **props):
        if mesh is None or len(mesh.faces) == 0:
            return None
        if id is None:
            key = f"{system}.{layer}".replace("/", ".")
            self._ids[key] += 1
            id = f"{key}.{self._ids[key]:04d}"
        e = Element(id, system, layer, ifc, material, mesh, level, props)
        self.elements.append(e)
        return e

    def curve(self, layer, points, **props):
        self.curves.append({"layer": layer, "points": [tuple(map(float, p)) for p in points], **props})

    def flag(self, msg):
        self.notes.append(msg)

    def by_system(self):
        d = defaultdict(list)
        for e in self.elements:
            d[e.system].append(e)
        return d

    def summary(self):
        c = Counter((e.system, e.layer) for e in self.elements)
        return sorted(c.items())
