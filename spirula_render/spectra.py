"""Light spectra for the lens flare and the diffraction gratings (from Vintage Click's
tools/make_spectra.py, imported into optics.json).

A light's colour (what lights the scan, and what Cycles uses for meshes) stays its Blender
colour; its spectrum says how that light is spread over the wavelengths, which is what a
grating fans out along its spikes and what the glass's dispersion bends ghost by ghost. A
tungsten lamp is strong in the red and weak in the blue, a white LED a blue spike and a broad
yellow hump, a fluorescent tube and a sodium or mercury lamp a few bright lines.
"""

import math

import numpy as np

from . import optics

LUMA = np.array([0.2126, 0.7152, 0.0722])

SPECTRUM_ITEMS = [
    ('COLOUR', "From Colour", "A smooth spectrum fitted to the light's colour"),
    ('TUNGSTEN', "Tungsten", "A hot filament: a black body at the light's temperature (3200 K for studio lamps)"),
    ('DAYLIGHT', "Daylight", "CIE D65: sun and sky, the closest to even"),
    ('LED', "White LED", "A blue LED under a yellow phosphor: a narrow blue spike, a dip in the cyan, a broad hump"),
    ('FLUORESCENT', "Fluorescent", "A tri-phosphor tube: narrow lines in the blue, green and orange-red over a faint glow"),
    ('SODIUM', "Sodium", "A high-pressure sodium street lamp: broad yellow-orange, a dark notch at 589 nm"),
    ('LOW_SODIUM', "Low-Pressure Sodium", "One yellow line at 589 nm: no rainbow at all"),
    ('MERCURY', "Mercury Vapour", "Mercury's lines: violet, blue, green and yellow"),
]


def data():
    return optics.data()["spectra"]


def wavelengths():
    return np.asarray(data()["wavelengths"], np.float64)


def dens():
    return np.asarray(data()["dens"], np.float64)


def rgb(S):
    return (np.asarray(S)[:, None] * dens()).sum(0)


def _line(wl, centre, width=4.0):
    """A spectral line spread over the 10 nm bins it falls in."""
    return np.exp(-0.5 * ((wl - centre) / width) ** 2) / (width * math.sqrt(2 * math.pi)) * 10


def planck(T, wl):
    lam = wl * 1e-9
    b = 1 / (lam ** 5 * (np.exp(1.438777e-2 / (lam * T)) - 1))
    return b / b[np.argmin(np.abs(wl - 560))]


def spectrum(kind, colour=(1, 1, 1), temperature=3200.0):
    """The light's spectrum over the 36 bins (its shape: scaled later to its own colour)."""
    wl = wavelengths()
    g = lambda c, s: np.exp(-0.5 * ((wl - c) / s) ** 2)
    if kind == 'TUNGSTEN':
        return planck(max(temperature, 1000.0), wl)
    if kind == 'DAYLIGHT':
        return np.asarray(data()["d65"]) / 100
    if kind == 'LED':
        return 1.2 * g(452, 11) + 0.75 * g(565, 55) + 0.04
    if kind == 'FLUORESCENT':
        return (0.05 + 0.04 * g(580, 80) + 1.75 * _line(wl, 436) + 2.4 * _line(wl, 545) + 1.6 * _line(wl, 611)
                + 0.6 * _line(wl, 487) + 0.4 * _line(wl, 405))
    if kind == 'SODIUM':
        return 0.03 + g(583, 38) * (1 - 0.85 * g(589, 6)) + 0.35 * _line(wl, 569) + 0.2 * _line(wl, 616) + 0.08 * _line(wl, 498)
    if kind == 'LOW_SODIUM':
        return 0.002 + _line(wl, 589, 3.0)
    if kind == 'MERCURY':
        return 0.02 + 0.6 * _line(wl, 405) + 1.3 * _line(wl, 436) + 1.6 * _line(wl, 546) + 1.0 * _line(wl, 578)
    # from colour: the smooth basis the game fits to a scene highlight's colour
    gr = data()["grating"]
    B = np.asarray(gr["basis"])
    w = np.asarray(gr["basis_inv"]) @ np.asarray(colour, np.float64)
    return np.maximum(w @ B, 0.0)


def colour_of(S):
    """The light colour a spectrum looks (brightest channel 1), for Match Colour."""
    c = np.maximum(rgb(S), 0.0)
    return c / max(c.max(), 1e-9)


def band_shares(S):
    """Each ghost band's share of the light's colour, per channel (sum 1 per channel)."""
    wl = wavelengths()
    D = dens()
    out = []
    for lo, hi in data()["bands"]:
        sel = (wl >= lo) & (wl <= hi)
        out.append(np.maximum((np.where(sel, S, 0.0)[:, None] * D).sum(0), 0.0))
    out = np.array(out)
    total = out.sum(0)
    return np.where(total > 1e-12, out / np.maximum(total, 1e-12), 1.0 / len(out))


def band_bend():
    """Each band's dispersion (+ toward the blue) and its span, in the same units."""
    return ([(550.0 - c) / 90.0 for c in data()["band_nm"]], list(data()["band_width"]))


def grating_lamp(u, S, I):
    """The colour along a grating's spike (u: share of its length, array) from a light of
    spectrum S and colour I: its spectrum spread order by order; hue from the spectrum,
    brightness from the light."""
    gr = data()["grating"]
    D = dens() * np.asarray(S)[:, None]
    u = np.asarray(u, np.float64)
    out = np.zeros(u.shape + (3,))
    for m, eta in enumerate(gr["eta"], 1):
        span = 730.0 / (m * gr["u1"])
        lam = u * span
        ok = (lam >= 380) & (lam <= 730)
        f = np.clip((lam - 380) / 10, 0, 34.999)
        i = np.minimum(f.astype(int), 34)
        t = (f - i)[..., None]
        d = D[i] * (1 - t) + D[i + 1] * t
        out += np.where(ok[..., None], eta * d * span / 10.0, 0.0)
    out = np.maximum(out, 0.0)
    lumS = max(float(rgb(S) @ LUMA), 1e-9)
    return out * (gr["k"] * float(np.asarray(I) @ LUMA) / lumS)


def grating_scene_matrix(u):
    """M(u) (3x3 per u): the colour along a grating's spike for a scene highlight of colour v
    is M(u) v (the game's table, interpolated)."""
    M = np.asarray(data()["grating"]["m"])          # (U, 3, 3)
    U = len(M)
    x = np.clip(np.asarray(u, np.float64) * U - 0.5, 0, U - 1.000001)
    i = np.minimum(x.astype(int), U - 2)
    t = (x - i)[..., None, None]
    return M[i] * (1 - t) + M[i + 1] * t


# ------------------------------------------------------------------ the glass's flaws
def dust_grade(lens):
    """How much dust and how many marks a lens's glass carries (0 clean, up to 3)."""
    c = lens.get("collection", "")
    if c == "Toys & Mods":
        return 3
    if c in ("Exotic", "Screw-Mount Classics", "Bokeh Monsters"):
        return 2
    return 1 if c else 0                              # a plain Blender camera: clean


def bubbles(lens):
    return any(lens.get("name", "").startswith(p) for p in ("Krasnogorsk", "Leningrad", "Jena"))
