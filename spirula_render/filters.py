"""Lens filters from Vintage Click (Filters.swift and the `filter_fx` shader).

A filter is screwed onto the front of the lens, so everything that reaches the film goes
through it. `transmission` is how much of each colour it passes (linear Rec.709): neutral
density, the contrast filters for black-and-white film, colour correction. The effect filters
also scatter some of the light:
- a cross screen's spikes off every bright point (4, 6 or 8 rays; a diffraction grating's split
  into rainbow bands, violet nearest);
- a soft-focus filter's glow round highlights and its blur mixed into the whole image;
- a mist or fog filter's veil lifting the shadows.

"Bright" is relative to the scene, as in the game: a point must be well above the frame's
average (five times it, between 0.35 and 0.95 of white, in exposed light) to throw spikes.

The game spreads its samples with a random offset per pixel; here the same kernels are
applied as their averages (the continuous ray and disc they sample), by FFT convolution.
"""

import math

import numpy as np

from . import optics

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def filters():
    return optics.data().get("filters", [])


def by_name(name):
    return next((f for f in filters() if optics.slug(f["name"]) == name), None)


def chosen(settings):
    """The filter in the engine's settings, or None (no filter, or none chosen)."""
    f = by_name(getattr(settings, "lens_filter", "NONE"))
    return f if f is not None and f["name"] != filters()[0]["name"] else None


def has_effect(f):
    return f["starPoints"] > 0 or f["glow"] > 0 or f["soften"] > 0 or f["veil"] > 0


def fits(f, lens):
    """A lens with a rear filter slot (a mirror lens, the widest) takes no effect filters."""
    return not (lens is not None and lens.get("rearFilter") and has_effect(f))


def stops(f):
    """The light it loses, in stops (what a through-the-lens meter would make up for)."""
    t = float(np.dot(f["transmission"], LUMA))
    return -math.log2(max(t, 1e-4))


def threshold(lin):
    """How bright a point must be to throw spikes or a halo: well above the scene's average."""
    meter = float(np.mean(lin @ LUMA)) if lin.size else 0.0
    return min(0.95, max(0.35, meter * 5))


def _above(v, thr):
    """Light brighter than `thr`, keeping its colour."""
    lum = v @ LUMA
    return v * (np.maximum(lum - thr, 0) / np.maximum(lum, 1e-4))[..., None]


def spectrum(h):
    """A spectrum colour, h from 0 (violet) to 1 (red), roughly even in brightness."""
    def ss(a, b, x):
        t = np.clip((x - a) / (b - a), 0, 1)
        return t * t * (3 - 2 * t)
    c = np.stack([ss(0.40, 0.85, h) + 0.4 * (1 - ss(0.0, 0.12, h)),
                  1 - np.clip(np.abs(h - 0.52) * 2.4, 0, 1),
                  1 - ss(0.08, 0.50, h)], -1)
    return np.clip(c, 0, 1) * 1.4


def _convolve(img, kernel):
    """img [H,W,C] (edges clamped) convolved with kernel [k,k] or [k,k,C] (odd k), by FFT."""
    H, W, C = img.shape
    k = kernel.shape[0]
    r = k // 2
    pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode='edge')
    sh = (pad.shape[0] + k - 1, pad.shape[1] + k - 1)
    fh, fw = 1 << (sh[0] - 1).bit_length(), 1 << (sh[1] - 1).bit_length()
    out = np.empty((H, W, C), np.float32)
    for c in range(C):
        kc = kernel if kernel.ndim == 2 else kernel[..., c]
        F = np.fft.rfft2(pad[..., c], (fh, fw)) * np.fft.rfft2(kc, (fh, fw))
        full = np.fft.irfft2(F, (fh, fw))
        out[..., c] = full[2 * r:2 * r + H, 2 * r:2 * r + W]
    return out


def _star_kernel(points, length, angle, spectral):
    """The rays of a cross screen: each line both ways, weight 3 exp(-3.2 sqrt(d/L)) / L per
    unit length out to L (what the game's samples at d = L t^2 average to). Spectral: one
    kernel per channel, the rainbow repeating along the ray."""
    r = int(math.ceil(length)) + 2
    k = 2 * r + 1
    ker = np.zeros((k, k, 3), np.float64)
    lines = max(1, points // 2)
    step = 0.25
    d = np.arange(step / 2, length, step)
    t = np.sqrt(d / length)
    w = 3.0 * np.exp(-3.2 * t) / length * step
    if spectral:
        h = np.mod(t * 3.2 + 0.15, 1.0)
        mix = np.clip((t - 0.02) / 0.08, 0, 1)
        mix = mix * mix * (3 - 2 * mix)
        col = (1 - mix)[:, None] + spectrum(h) * mix[:, None]
        wc = w[:, None] * col * 1.3
    else:
        wc = np.repeat(w[:, None], 3, 1)
    for j in range(lines):
        a = angle + j * math.pi / lines
        for side in (-1, 1):
            x = r + side * d * math.cos(a)
            y = r + side * d * math.sin(a)
            x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
            fx, fy = x - x0, y - y0
            for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                               (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                for c in range(3):
                    np.add.at(ker[..., c], (y0 + dy, x0 + dx), wc[:, c] * ww)
    return ker.astype(np.float32)


def _disc_kernel(radius):
    """The game's halo samples, radius t^2 R over a disc (t even): mass inside r is sqrt(r/R),
    so it's bright near the middle and fades with no edge. Built from the ring masses."""
    r = int(math.ceil(radius)) + 1
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    # average the radial density over each pixel with a few sub-samples
    acc = np.zeros_like(x)
    for sy in (-0.375, -0.125, 0.125, 0.375):
        for sx in (-0.375, -0.125, 0.125, 0.375):
            rr = np.hypot(x + sx, y + sy)
            dens = np.where(rr < radius, 1.0 / (4 * math.pi * np.sqrt(np.maximum(rr, 0.15) * radius)
                                                 * np.maximum(rr, 0.15)), 0.0)
            acc += dens
    ker = acc / 16
    return (ker / ker.sum()).astype(np.float32)


def apply(lin, f, film_stops=0.0, lens=None, vig=None):
    """The filter on the frame: lin [H,W,3] linear, exposed light. Returns the filtered frame
    (transmission, then the effects' light added and taken). `film_stops`: the film exposure
    still to come, so the threshold is judged in the light the film gets."""
    lin = lin * np.asarray(f["transmission"], np.float32)
    if not has_effect(f) or not fits(f, lens):
        return lin
    H, W = lin.shape[:2]
    g = np.float32(2.0 ** film_stops)
    src = lin * g
    thr = threshold(src)
    add = np.zeros_like(src)
    if f["starPoints"] > 0:
        spectral = bool(f["spectral"])
        L = H * (0.20 if spectral else 0.15)
        hi = _above(src, thr)
        ker = _star_kernel(int(f["starPoints"]), L, 0.35, spectral)
        if spectral:
            lum = (hi @ LUMA)[..., None]
            add += _convolve(np.repeat(lum, 3, -1), ker) * f["starStrength"]
        else:
            add += _convolve(hi, ker) * f["starStrength"]
    if f["glow"] > 0 or f["soften"] > 0 or f["veil"] > 0:
        gthr = thr * 0.45
        kn, kf = _disc_kernel(H * 0.02), _disc_kernel(H * 0.09)
        hi = _above(src, gthr)
        if f["glow"] > 0:
            add += f["glow"] * (_convolve(hi, kn) * 1.2 + _convolve(hi, kf) * 1.6)
        if f["soften"] > 0:
            add += f["soften"] * (_convolve(src, kn) - src)
        if f["veil"] > 0:
            far = _convolve(src, kf)
            add += f["veil"] * ((far - src) * 0.6 + far * 0.12)
    if vig is not None:
        add = add * vig[..., None]
    return np.maximum(lin + add / g, 0.0)
