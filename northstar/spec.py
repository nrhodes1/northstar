"""Load the spec/ folder into one object with convenience accessors."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .units import L

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "spec"


@dataclass
class Spec:
    data: dict = field(default_factory=dict)
    root: Path = SPEC_DIR

    def __getitem__(self, k):
        return self.data[k]

    def get(self, k, d=None):
        return self.data.get(k, d)

    # ---- frame -------------------------------------------------------
    @property
    def levels(self) -> dict[str, float]:
        lv = self.data["project"]["levels"]
        return {k: L(v) for k, v in sorted(lv.items(), key=lambda kv: L(kv[1]))}

    def level_z(self, name: str) -> float:
        return self.levels[name]

    def level_above(self, name: str) -> str | None:
        names = list(self.levels)
        i = names.index(name)
        return names[i + 1] if i + 1 < len(names) else None

    def grid(self, axis: str, name: str) -> float:
        return L(self.data["project"]["grids"][axis][str(name)])

    def storeys(self) -> list[str]:
        return list(self.data["massing"]["storeys"].keys())


def load(spec_dir: Path | str = SPEC_DIR) -> Spec:
    spec_dir = Path(spec_dir)
    data = {}
    for p in sorted(spec_dir.glob("*.yaml")):
        with open(p) as f:
            data[p.stem] = yaml.safe_load(f) or {}
    # sub-folders (e.g. spec/interiors/L1.yaml) -> data["interiors"]["L1"]
    for d in sorted(x for x in spec_dir.iterdir() if x.is_dir()):
        data[d.name] = {}
        for p in sorted(d.glob("*.yaml")):
            with open(p) as f:
                data[d.name][p.stem] = yaml.safe_load(f) or {}
    return Spec(data=data, root=spec_dir)
