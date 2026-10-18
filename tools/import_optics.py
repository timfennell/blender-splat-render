"""Read the lenses and film stocks from Splat Camera's Swift source into optics.json.

    python tools/import_optics.py [path/to/splat_camera]

Splat Camera (the vintage camera game) is the master copy: its Lens.swift and FilmStock.swift
hold the camera bag and the film fridge as literal lists. This reads those literals (it is not
a Swift parser: it knows the shape of `Lens(name: ..., focalMM: ...)` and `FilmStock(...)`
calls), fills in the struct defaults, and writes spirula_render/optics.json. Re-run it after
changing a lens or a stock in the game. Fisheyes are left out (the engine draws perspective
views only).
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "spirula_render", "optics.json")

LENS_DEFAULTS = dict(lateralCA=0.0005, axialCA=0.0, projection="rectilinear", obstruction=0.0,
                     sphericalAberration=0.0, squeeze=1.0, curvatureRadial=0.0,
                     curvatureTangential=0.0, movement=None, streak=0.0)
FILM_DEFAULTS = dict(blackAndWhite=False, bwMix=[0.30, 0.59, 0.11],
                     crosstalk=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], balance=[1, 1, 1],
                     saturation=1.0, contrast=1.0, toe=1.0, shoulder=1.0, blackLift=0.02,
                     grain=0.3, grainSize=1.2, halation=0.0, halationTint=[1.0, 0.25, 0.05],
                     falseColourIR=False)


def calls(src, name):
    """Bodies of every `name(...)` call (balanced parentheses, strings respected)."""
    out = []
    for m in re.finditer(r"\b" + name + r"\(", src):
        i, depth, in_str = m.end(), 1, False
        while depth:
            ch = src[i]
            if in_str:
                if ch == "\\":
                    i += 1
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            i += 1
        out.append(src[m.end():i - 1])
    return out


def split_args(body):
    """Top-level `key: value` pairs of a call body (comments dropped)."""
    body = re.sub(r"//[^\n]*", "", body)
    parts, depth, in_str, cur = [], 0, False, ""
    for ch in body:
        if in_str:
            cur += ch
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    out = {}
    for p in parts:
        k, _, v = p.partition(":")
        out[k.strip()] = v.strip()
    return out


def value(v):
    v = v.strip()
    if v.startswith('"'):
        return json.loads(v)
    if v in ("true", "false"):
        return v == "true"
    if v.startswith("mix("):
        return [value(r) for r in re.findall(r"\[[^\]]*\]", v)]
    if v.startswith("["):
        return [float(x) for x in v.strip("[]").split(",") if x.strip()]
    m = re.fullmatch(r"\.(tilt|shift)\(([-\d.]+)\)", v)
    if m:
        return {"kind": m.group(1), "most": float(m.group(2))}
    if v.startswith("."):
        return {"fisheye": "fisheye", "circularFisheye": "circular_fisheye"}.get(v[1:], v[1:])
    return float(v)


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Documents/splat_camera")
    lens_src = open(os.path.join(game, "Sources/SplatCamera/Lens.swift")).read()
    film_src = open(os.path.join(game, "Sources/SplatCamera/FilmStock.swift")).read()

    # which list each lens is in, and the lists' collection names
    shelves = {}
    for m in re.finditer(r"LensCollection\(name: \"([^\"]+)\", about: \"([^\"]+)\", lenses: (\w+)\)", lens_src):
        shelves[m.group(3)] = (m.group(1), m.group(2))
    lenses, collections = [], []
    for m in re.finditer(r"private let (\w+): \[Lens\] = \[", lens_src):
        var = m.group(1)
        end = lens_src.index("\n]", m.end())
        coll, about = shelves.get(var, (var, ""))
        collections.append({"name": coll, "about": about})
        for body in calls(lens_src[m.end():end], "Lens"):
            d = dict(LENS_DEFAULTS)
            d.update({k: value(v) for k, v in split_args(body).items()})
            d["collection"] = coll
            if d["projection"] != "rectilinear":
                continue                     # fisheyes: not in the Blender engine
            lenses.append(d)
    # keep the game's shelf order
    order = [s[0] for s in shelves.values()]
    collections.sort(key=lambda c: order.index(c["name"]) if c["name"] in order else 99)
    lenses.sort(key=lambda l: order.index(l["collection"]) if l["collection"] in order else 99)

    films = []
    start = film_src.index("let filmStocks: [FilmStock] = [")
    for body in calls(film_src[start:], "FilmStock"):
        d = dict(FILM_DEFAULTS)
        d.update({k: value(v) for k, v in split_args(body).items()})
        films.append(d)

    json.dump({"source": "Splat Camera (Lens.swift, FilmStock.swift)", "collections": collections,
               "lenses": lenses, "films": films}, open(OUT, "w"), indent=1)
    print(f"{len(lenses)} lenses in {len(collections)} collections, {len(films)} film stocks -> {OUT}")


if __name__ == "__main__":
    main()
