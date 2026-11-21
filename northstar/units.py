"""Length parsing. Spec values are feet (numbers) or strings with units.

Internally everything is METRES.
"""
from __future__ import annotations

import re
from fractions import Fraction

FT = 0.3048
IN = 0.0254

_num = r"(\d+(?:\.\d+)?(?:\s+\d+/\d+)?|\d+/\d+)"


def _frac(s: str) -> float:
    s = s.strip()
    total = 0.0
    for part in s.split():
        total += float(Fraction(part))
    return total


def L(v) -> float:
    """Parse a length into metres.

    Numbers are feet. Strings: "11'-6\"", "3/4\"", "1 1/2\"", "2'", "400mm",
    "0.4m", "12in", "-12'-4\"".
    """
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v) * FT
    s = str(v).strip().replace("’", "'").replace("”", '"')
    sign = 1.0
    if s.startswith("-"):
        sign, s = -1.0, s[1:].strip()
    m = re.fullmatch(r"([\d.]+)\s*(mm|cm|m|in|ft)", s)
    if m:
        val, u = float(m.group(1)), m.group(2)
        return sign * val * {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": IN, "ft": FT}[u]
    m = re.fullmatch(_num + r"\s*'\s*(?:-?\s*" + _num + r'\s*")?', s)
    if m:
        ft = _frac(m.group(1))
        inch = _frac(m.group(2)) if m.group(2) else 0.0
        return sign * (ft * FT + inch * IN)
    m = re.fullmatch(_num + r'\s*"', s)
    if m:
        return sign * _frac(m.group(1)) * IN
    try:
        return sign * float(s) * FT
    except ValueError as e:  # pragma: no cover
        raise ValueError(f"Cannot parse length {v!r}") from e


def P(pt) -> tuple:
    """[x_ft, y_ft] (or with z) -> metres tuple."""
    return tuple(L(c) for c in pt)


def fmt_ftin(m: float) -> str:
    """Metres -> 12'-4 1/2\" (nearest 1/16")."""
    neg = m < 0
    inches = round(abs(m) / IN * 16) / 16
    ft = int(inches // 12)
    rem = inches - ft * 12
    whole = int(rem)
    frac = Fraction(rem - whole).limit_denominator(16)
    s = f"{ft}'-{whole}"
    if frac:
        s += f" {frac.numerator}/{frac.denominator}"
    s += '"'
    return ("-" if neg else "") + s
