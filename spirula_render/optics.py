"""Vintage lenses and film stocks, as Vintage Click (the camera game, formerly Splat Camera) defines them.

optics.json is generated from the game's source by tools/import_optics.py. The maths here are
ports of the game's (CameraBody.swift, RenderLoop.swift, Shaders.swift): the pupil (blades,
barrel clipping into cat's eyes, rim-bright or rim-dark bokeh, a mirror lens's ring, an oval
stop), the lens table the renderer's projection takes (sr_set_lens: off-axis pupil moments,
tilt and field curvature, spherical aberration), the sub-pupil "cells" an exposure adds up,
and development: fall-off, lateral colour, streak flare, halation, the stock's curve, colour
and grain.
"""

import json
import math
import os

import numpy as np

_DATA = None


def data():
    global _DATA
    if _DATA is None:
        with open(os.path.join(os.path.dirname(__file__), "optics.json")) as f:
            _DATA = json.load(f)
    return _DATA


def lenses():
    return data()["lenses"]


def films():
    return data()["films"]


def lens_by_name(name):
    return next((l for l in lenses() if l["name"] == name), None)


def film_by_name(name):
    return next((f for f in films() if f["name"] == name), None)


def slug(name):
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


def chosen(settings):
    """(lens, film) chosen in the engine's settings, each None when off."""
    lens = next((l for l in lenses() if slug(l["name"]) == getattr(settings, "lens", "NONE")), None)
    film = next((f for f in films() if slug(f["name"]) == getattr(settings, "film", "NONE")), None)
    return lens, film


def aperture_stops(lens):
    full = [1.0, 1.4, 2, 2.8, 4, 5.6, 8, 11, 16, 22, 32]
    stops = [s for s in full if lens["maxAperture"] * 1.02 < s <= lens["minAperture"] * 1.02]
    return [lens["maxAperture"]] + stops


# ---------------------------------------------------------------- the pupil

def pupil_weight(l, q, lens, n):
    """Light through lens point(s) l [..,2] (unit = the current aperture's radius) seen from
    field position(s) q [..,2] (0 centre, 1 frame corner): 0 outside the blades, in a mirror
    lens's obstruction or clipped by the barrel, else the rim weighting. (pupilWeight)"""
    l = np.asarray(l, np.float64)
    q = np.asarray(q, np.float64)
    r = np.linalg.norm(l, axis=-1)
    blades = int(lens["blades"])
    if blades >= 3:
        seg = 2 * math.pi / blades
        th = np.arctan2(l[..., 1], l[..., 0]) - lens["bladeRotation"]
        a = np.mod(np.mod(th, seg) + seg, seg) - seg / 2
        ok = ~(r * np.cos(a) > math.cos(math.pi / blades))
    else:
        ok = r <= 1
    ok = ok & (r >= lens["obstruction"])
    opn = lens["maxAperture"] / n
    ok = ok & (np.linalg.norm(l * opn + lens["catsEye"] * q, axis=-1) <= 1)
    rim = (1 + lens["soapBubble"] * r ** 6) / (1 + lens["soapBubble"] * 0.25)
    return np.where(ok, rim, 0.0)


def rim_weight(l, lens):
    """The rim weighting alone (a cell already clipped by the barrel in the render)."""
    r = np.linalg.norm(np.asarray(l, np.float64), axis=-1)
    return (1 + lens["soapBubble"] * r ** 6) / (1 + lens["soapBubble"] * 0.25)


def ring_cells(n, turn):
    """`n` lens points of equal area in rings over the unit disc, the outer ring on the rim."""
    rings = max(1, int(round(math.sqrt(n / math.pi))))
    radii = [(i + 0.5) / rings for i in range(rings)]
    total = sum(radii)
    counts = [max(1, int(round(n * r / total))) for r in radii]
    counts[-1] = max(1, counts[-1] + n - sum(counts))
    out = []
    for i, r in enumerate(radii):
        a0 = turn + i * 2.39996323
        for j in range(counts[i]):
            a = a0 + 2 * math.pi * j / counts[i]
            out.append((r * math.cos(a), r * math.sin(a)))
    return out[:n]


def splits_colour(lens, cells):
    return lens["axialCA"] > 0 and cells >= 64


def cell_centres(lens, k):
    """The exposure's sub-pupils (aperture radii), stretched to the blades' polygon; with
    axial colour every third carries one channel, each channel a whole set of its own."""
    split = splits_colour(lens, k)
    sets = ([ring_cells((k - c + 2) // 3, c * 2.1) for c in range(3)] if split
            else [ring_cells(k, 0.0)])
    out = []
    blades = int(lens["blades"])
    for i in range(k):
        v = np.array(sets[i % 3][i // 3] if split else sets[0][i], np.float64)
        if blades >= 3:
            seg = 2 * math.pi / blades
            th = math.atan2(v[1], v[0])
            a = math.fmod(math.fmod(th - lens["bladeRotation"], seg) + seg, seg) - seg / 2
            v *= math.cos(math.pi / blades) / math.cos(a)
        out.append(v)
    return out


TABLE = 32


def pupil_table(lens, n):
    """Pupil moments against field radius: (mean along q, variance along q, variance across
    q, light passed), q from 0 to 1.25. (CameraBody.pupilTable)"""
    g = 72
    c = (np.arange(g) + 0.5) / g * 2 - 1
    lx, ly = np.meshgrid(c, c)
    L = np.stack([lx.ravel(), ly.ravel()], 1)
    rows = []
    centre = None
    for i in range(TABLE):
        q = np.array([i / (TABLE - 1) * 1.25, 0.0])
        p = pupil_weight(L, q[None, :], lens, n)
        w = p.sum()
        if i == 0:
            centre = max(w, 1e-6)
        if w <= 0:
            rows.append((0, 0, 0, 0))
            continue
        mx, my = (p * L[:, 0]).sum() / w, (p * L[:, 1]).sum() / w
        rows.append((mx, (p * L[:, 0] ** 2).sum() / w - mx * mx,
                     (p * L[:, 1] ** 2).sum() / w - my * my, w / centre))
    return rows


def lens_params(lens, n, focal_px, cells, axis, field_radius, metres_per_unit, focus_m,
                tilt=(0.0, 0.0), table=None, cell=(0.0, 0.0), patch=0.0):
    """The float4 table sr_set_lens takes (layout: the spirula fork's
    docs/notes/lens-defocus.md). `cells` > 1: one sub-pupil, blurred by only its own spread;
    1 with a table: the whole pupil in one render (a live preview). (CameraBody.lensParams)"""
    radius_units = lens["focalMM"] / 1000 / n / 2 / metres_per_unit
    inv_focus = metres_per_unit / focus_m
    p = [(1, inv_focus, radius_units, 1 if cells == 1 and table is not None else 0),
         (1 / (2 * cells), 1, 0, focal_px * 0.25)]
    p += list(table or [])
    p += [(0, 0, 0, 0)] * (TABLE - len(table or []))
    f = lens["focalMM"] / 1000
    per_mm = 0.001 / (f * f) * metres_per_unit
    half_diag = 21.63
    deg = math.pi / 180
    p.append((math.tan(tilt[0] * deg) * half_diag * per_mm, -math.tan(tilt[1] * deg) * half_diag * per_mm,
              lens["curvatureRadial"] * per_mm, lens["curvatureTangential"] * per_mm))
    opn = lens["maxAperture"] / n
    rim = lens["sphericalAberration"] * opn * opn * metres_per_unit if cells > 1 else 0
    p.append((cell[0], cell[1], lens["squeeze"], rim))
    p.append((axis[0], axis[1], field_radius, 0))
    p.append((lens["catsEye"], opn, patch if cells > 1 else 0, 0))
    return np.asarray(p, np.float32)


def field_focus(lens, metres_per_unit, tilt, qx, qy):
    """Extra 1/focus (scene units) over the field: tilt and the mean of radial and
    tangential curvature -- what a mesh pixel gets (the splats get both, separately)."""
    f = lens["focalMM"] / 1000
    per_mm = 0.001 / (f * f) * metres_per_unit
    deg = math.pi / 180
    tx = math.tan(tilt[0] * deg) * 21.63 * per_mm
    ty = -math.tan(tilt[1] * deg) * 21.63 * per_mm
    curv = 0.5 * (lens["curvatureRadial"] + lens["curvatureTangential"]) * per_mm
    return tx * qx + ty * qy + curv * (qx * qx + qy * qy)


# ---------------------------------------------------------------- focusing close

def magnification(lens, focus_m):
    """The lens's magnification focused at `focus_m` (thin lens, m = f / (s - f), blended near
    the closest focus toward the published figure: a macro's 1:2 or 1:1). (CameraBody)"""
    f = lens["focalMM"] / 1000
    focus_m = max(focus_m, lens["minFocusM"])          # the focus ring stops at the closest focus
    simple = f / max(focus_m - f, f * 0.05)
    if lens.get("maxMagnification", 0) <= 0:
        return simple
    at_closest = f / max(lens["minFocusM"] - f, f * 0.05)
    k = lens["maxMagnification"] / at_closest
    t = min(1.0, lens["minFocusM"] / max(focus_m, lens["minFocusM"]))
    return simple * (1 + (k - 1) * t * t)


def close_focus_loss(lens, focus_m):
    """Stops lost focusing close (the bellows factor: effective aperture N (1 + m))."""
    return 2 * math.log2(1 + magnification(lens, focus_m))


def breathing_scale(lens, focus_m):
    """How much the picture grows focusing close: 1 + m as the whole lens moves out (focus
    breathing); an internal-focus lens (breathing < 0) widens a little instead."""
    return max(0.5, 1 + lens.get("breathing", 1.0) * magnification(lens, focus_m))


# ---------------------------------------------------------------- film

def developed(stock, push):
    """The stock as the lab develops it after rating it `push` stops off box speed."""
    s = dict(stock)
    if push == 0:
        return s
    x = float(push)
    if push > 0:
        s["contrast"] *= 1 + 0.12 * x
        s["toe"] *= 1 + 0.06 * x
        s["shoulder"] *= 1 + 0.08 * x
        s["grain"] *= 1 + 0.3 * x
        s["blackLift"] += 0.006 * x
        s["saturation"] *= 1 + 0.04 * x
    else:
        s["contrast"] *= 1 + 0.10 * x
        s["shoulder"] *= 1 + 0.08 * x
        s["grain"] *= 1 + 0.15 * x
    return s


def _hash(px, py, s):
    h = (px.astype(np.uint32) * np.uint32(0x8da6b343)) ^ (py.astype(np.uint32) * np.uint32(0xd8163841)) \
        ^ np.uint32((s * 0xcb1ab31f) & 0xffffffff)
    h ^= h >> np.uint32(13)
    h *= np.uint32(0x5bd1e995)
    h ^= h >> np.uint32(15)
    return (h & np.uint32(0xffffff)).astype(np.float32) / 16777215.0


def _value_noise(x, y, s):
    ix, iy = np.floor(x), np.floor(y)
    tx, ty = x - ix, y - iy
    tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
    ax, ay = (ix + 65536).astype(np.int64), (iy + 65536).astype(np.int64)
    v00, v10 = _hash(ax, ay, s), _hash(ax + 1, ay, s)
    v01, v11 = _hash(ax, ay + 1, s), _hash(ax + 1, ay + 1, s)
    top = v00 + (v10 - v00) * tx
    bot = v01 + (v11 - v01) * tx
    return top + (bot - top) * ty


def film_look(lin, halo, stock, seed, grain=True):
    """Scene-linear exposure [H,W,3] (mid grey 0.18 = correctly exposed) to a print, display
    values 0..1. (film_look in the game's Shaders.swift)"""
    lin = np.asarray(lin, np.float32)
    H, W = lin.shape[:2]
    if stock["halation"] > 0 and halo is not None:
        lin = lin + stock["halation"] * np.asarray(stock["halationTint"], np.float32) * halo
    if stock["falseColourIR"]:
        lum = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
        leaf = np.clip(3.0 * (g - np.maximum(r, b)) / np.maximum(g, 1e-4), 0, 1)
        sky = np.clip((b - np.maximum(r, g)) / np.maximum(b, 1e-4), 0, 1)
        reds = np.clip((r - np.maximum(g, b)) / np.maximum(r, 1e-4), 0, 1)
        ir = lum * (1 + 3.0 * leaf) * (1 - 0.75 * sky) + 0.9 * reds * r
        c = lin - lum[..., None]
        x = np.linalg.norm(c, axis=-1) / np.maximum(lum, 1e-4)
        t = np.clip((x - 0.06) / 0.24, 0, 1)
        strength = t * t * (3 - 2 * t)
        sw = np.stack([ir - lum, c[..., 0] * strength,
                       c[..., 1] * strength - 0.45 * leaf * g + 0.35 * sky * b], -1)
        lin = np.maximum(lum[..., None] + sw, 0) * (1 - 0.3 * sky)[..., None]
    M = np.asarray(stock["crosstalk"], np.float32)
    c = (lin @ M.T) * np.asarray(stock["balance"], np.float32)
    bw = bool(stock["blackAndWhite"])
    if bw:
        c = np.repeat((c @ np.asarray(stock["bwMix"], np.float32))[..., None], 3, -1)
    luma = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    c = np.maximum(luma[..., None] + (c - luma[..., None]) * stock["saturation"], 0)
    x = np.log2(np.maximum(c, 1e-6) / 0.18)
    k = stock["contrast"] * 0.34
    xs = np.where(x > 0, x * stock["shoulder"], x * stock["toe"])
    y = 0.5 + 0.5 * np.tanh(k * xs)
    y = stock["blackLift"] + (1 - stock["blackLift"]) * y
    if grain and stock["grain"] > 0:
        size = stock["grainSize"] * H / 1080.0
        py, px = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5
        gx, gy = px / max(size, 0.5), py / max(size, 0.5)
        amp = stock["grain"] * math.sqrt(stock["iso"] / 100) * 0.16
        mono = _value_noise(gx, gy, seed) + _value_noise(gx * 2.1 + 17, gy * 2.1 + 17, seed + 1) * 0.5 - 0.75
        if bw:
            nz = mono[..., None]
        else:
            chroma = np.stack([_value_noise(gx + 31, gy + 31, seed + 2), _value_noise(gx + 57, gy + 57, seed + 3),
                               _value_noise(gx + 83, gy + 83, seed + 4)], -1) - 0.5
            nz = mono[..., None] + 0.35 * chroma
        y = y + amp * nz * 4 * y * (1 - y)
    return np.clip(y, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- finishing the frame

def _sample_bilinear(img, x, y):
    H, W = img.shape[:2]
    x = np.clip(x - 0.5, 0, W - 1)
    y = np.clip(y - 0.5, 0, H - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    a, b = img[y0, x0], img[y0, x1]
    c, d = img[y1, x0], img[y1, x1]
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy


def lens_finish(lin, lens, n, axis):
    """Light fall-off toward the corners (stronger wide open) and lateral colour (red and
    blue at slightly different magnifications), on the linear image [H,W,3]."""
    H, W = lin.shape[:2]
    py, px = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5
    cx, cy = axis
    half = math.hypot(W / 2, H / 2)
    q2 = ((px - cx) ** 2 + (py - cy) ** 2) / (half * half)
    vig = 1 - lens["vignette"] * min(1.0, lens["maxAperture"] / n) ** 1.5 * q2
    k = lens["lateralCA"]
    out = np.empty_like(lin)
    out[..., 0] = _sample_bilinear(lin[..., 0:1], cx + (px - cx) * (1 - k), cy + (py - cy) * (1 - k))[..., 0]
    out[..., 1] = lin[..., 1]
    out[..., 2] = _sample_bilinear(lin[..., 2:3], cx + (px - cx) * (1 + k), cy + (py - cy) * (1 + k))[..., 0]
    return out * np.maximum(vig, 0)[..., None]


def streak(lin, share):
    """A thin wire across the stop diffracts `share` of the light into horizontal streaks
    (four exponential tails, a little longer in red), as the game's `streak` kernel."""
    H, W = lin.shape[:2]
    v = lin + np.maximum(lin - 0.85, 0) * 12          # clipped highlights stand for brighter lights
    out = np.zeros_like(lin)
    base = W * 0.004
    for j in range(4):
        length = base * 4 ** j * np.array([1.15, 1.0, 0.85], np.float32)
        a = np.exp(-1.0 / length).astype(np.float32)
        norm = (1 - a) / (1 + a) * (share / 4)
        run = np.zeros((H, 3), np.float32)
        fwd = np.empty_like(lin)
        for i in range(W):
            run = v[:, i] + a * run
            fwd[:, i] = run
        run[:] = 0
        bwd = np.empty_like(lin)
        for i in range(W - 1, -1, -1):
            run = v[:, i] + a * run
            bwd[:, i] = run - v[:, i]
        out += norm * (fwd + bwd)
    return out


def halation_light(lin):
    """Light past the film base bouncing back around bright highlights (32 taps over a disc
    of 1.2% of the frame height), as the game's `develop` gathers it."""
    H, W = lin.shape[:2]
    rad = H * 0.012
    over = np.maximum(lin - 0.9, 0)
    halo = np.zeros_like(lin)
    for j in range(32):
        r = math.sqrt((j + 0.5) / 32) * rad
        th = j * 2.39996323
        dx, dy = int(round(r * math.cos(th))), int(round(r * math.sin(th)))
        ys = np.clip(np.arange(H) + dy, 0, H - 1)
        xs = np.clip(np.arange(W) + dx, 0, W - 1)
        halo += over[ys][:, xs]
    return halo / 32


def srgb_to_linear(c):
    c = np.maximum(c, 0.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def develop(lin, stock, push, stops, seed, grain=True, halation=True):
    """The finished linear frame [H,W,3] printed on `stock` (pushed `push` stops, exposed
    `stops` over the engine's own exposure); returns linear values whose display (the
    Standard view transform) is the print."""
    s = developed(stock, push)
    lin = lin * np.float32(2.0 ** stops)
    halo = halation_light(lin) if halation and s["halation"] > 0 else None
    return srgb_to_linear(film_look(lin, halo, s, seed, grain))
