"""Lens flare from the scene's lights, ported from Vintage Click (Flare.swift and the
`lens_flare` shader).

Each light with Lens Flare on that shines into the lens adds, in frame light before the
film:
- its lit face (a point or spot light's sphere, an area light's panel, the sun's disc) at its
  true brightness, spread by its own defocus, and the bloom a lens scatters round it;
- veiling glare, a broad haze strongest near the light;
- ghosts: the iris's shape (round wide open, the blades' polygon stopped down) strung along
  the line through the axis, about 1e-4 of the light each for a single-coated lens and 1e-5
  multi-coated, each colour bent its own amount, softer away from the axis, each with its own
  best focus;
- a faint ring (a reflection off the film and back);
- a starburst from the iris's corners (none wide open, longer as it closes; 2n spikes for odd
  blade counts), only for a light in focus.

It works wavelength by wavelength from each light's spectrum (spectra.py): ghosts and the ring
split the light into nine bands, each bent by the glass's dispersion; a grating lays the
spectrum along its spikes. And old glass shows in the ghosts: dust, cleaning marks, coating
scratches, fungus, seed bubbles (glass_flaws), as Vintage Click draws them.

Brightness is physical, in the relighting's units: a white surface lit by a light renders as
that light's relative irradiance there, so the flare is what that light sends into the lens,
spread as the lens spreads a small part of it. A light hidden behind splats or meshes (seen
through the depth of the finished frame) flares only as much as it shows, and its face is
hidden pixel by pixel behind whatever stands in front of it.
"""

import math

import numpy as np

from . import spectra

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

# A Blender camera without a lens from the bag: a modern multi-coated lens, its iris from the
# camera's own aperture blades.
GENERIC = dict(name="", collection="", blades=0, bladeRotation=0.0, maxAperture=1.4,
               lateralCA=0.0005, obstruction=0.0, squeeze=1.0, curvatureRadial=0.0,
               curvatureTangential=0.0, sphericalAberration=0.0, focalMM=50.0)


def single_coated(lens):
    """Older and odd optics are single-coated: brighter, more colourful ghosts."""
    return lens.get("collection") in ("Exotic", "Toys & Mods") or "Petzval" in lens["name"] \
        or "Holga" in lens["name"]


def _seed(name):
    s = 2166136261
    for c in name.encode():
        s = ((s ^ c) * 16777619) & 0xFFFFFFFF
    return float(s % 10000)


def _hash(s, k):
    # as the shader computes it, in single precision
    a = np.float32(np.float32(s) * np.float32(12.9898) + np.float32(k) * np.float32(78.233))
    v = np.float32(np.float32(math.sin(float(a))) * np.float32(43758.5453))
    return float(v - np.floor(v))


def lens_params(lens, n, lens_filter=None):
    """The flare's lens: iris, coating, dispersion, field, the glass's flaws; and the filter on
    it, if any (flat glass adds a ghost of its own; a cross screen or grating, its spikes)."""
    f = lens_filter
    seed = _seed(lens["name"])
    dust = spectra.dust_grade(lens)
    return dict(dust=dust, bubbles=spectra.bubbles(lens), fungus=dust >= 2 and _hash(seed, 1300) < 0.55,
                glass=f is not None, star_points=int(f["starPoints"]) if f else 0,
                star_strength=float(f["starStrength"]) if f else 0.0,
                spectral=bool(f["spectral"]) if f else False,
                blades=int(lens["blades"]), rotation=float(lens["bladeRotation"]),
                open=min(1.0, float(lens["maxAperture"]) / n), single=single_coated(lens),
                seed=seed, obstruction=float(lens.get("obstruction", 0.0)),
                squeeze=float(lens.get("squeeze", 1.0)),
                dispersion=0.008 + float(lens["lateralCA"]) * 4,
                field=0.15 + (abs(lens.get("curvatureRadial", 0.0)) + abs(lens.get("curvatureTangential", 0.0)))
                * 0.6 + abs(lens.get("sphericalAberration", 0.0)) * 0.3)


# ------------------------------------------------------------------ the lights
def sources(lights, cam_world, fx, fy, cx, cy, W, H, gain, focal_mm=None, n=None, focus_m=None,
            metres_per_unit=1.0):
    """The flaring lights as the camera sees them. `lights`: dicts from the engine (type, pos,
    forward, right, up, colour, energy, radius, size, size_y, round, spot_cos, angle).
    focal_mm / n / focus_m give each light its own defocus blur (leave out for a pinhole)."""
    C = np.array(cam_world, np.float64)
    V = np.linalg.inv(C)
    Vr = np.diag([1.0, -1.0, -1.0]) @ V[:3, :3]         # world -> OpenCV camera axes
    cam = C[:3, 3]
    diag = math.hypot(W, H)
    omega = (W / fx) * (H / fy)                           # the frame's solid angle

    def project(q):
        c = Vr @ (np.asarray(q, np.float64) - cam)
        if c[2] <= 1e-9:
            return None
        return np.array([cx + fx * c[0] / c[2], cy + fy * c[1] / c[2]])

    out = []
    for L in lights:
        colour = np.asarray(L["colour"], np.float64)
        fwd = np.asarray(L["forward"], np.float64)
        if L["type"] == 'SUN':
            c = Vr @ (-fwd)
            if c[2] <= 1e-6:
                continue
            px = np.array([cx + fx * c[0] / c[2], cy + fy * c[1] / c[2]])
            e = gain * L["energy"] * colour
            a = max(L.get("angle", 0.0) / 2, 0.0047)       # the disc's angular radius
            rf = fx * math.tan(a)
            face = e / (a * a)                            # pi x its radiance (E = L pi a^2)
            dist, blur = math.inf, 0.0
            A, B = np.array([rf, 0.0]), np.array([0.0, fy * math.tan(a)])
            round_face = True
        else:
            pos = np.asarray(L["pos"], np.float64)
            rel = pos - cam
            dist = float(np.linalg.norm(rel))
            px = project(pos) if dist > 1e-9 else None
            if px is None:
                continue
            to_cam = -rel / dist
            ct = float(to_cam @ fwd)
            I = gain * L["energy"] / (4 * math.pi)        # candela, as the relighting has it
            if L["type"] == 'SPOT':
                cos_out, cos_in = L["spot_cos"]
                I *= min(max((ct - cos_out) / max(cos_in - cos_out, 1e-6), 0.0), 1.0)
            if L["type"] == 'AREA' and ct <= 0:
                continue                                  # seen from behind
            if I <= 0:
                continue
            e = I / (dist * dist) * colour
            if L["type"] == 'AREA':
                right, up = np.asarray(L["right"]), np.asarray(L["up"])
                sx, sy = L["size"] / 2, L["size_y"] / 2
                pa, pb = project(pos + right * sx), project(pos + up * sy)
                A = (pa - px) if pa is not None else np.zeros(2)
                B = (pb - px) if pb is not None else np.zeros(2)
                round_face = L["round"]
                area = math.pi * sx * sy if round_face else 4 * sx * sy
                face = math.pi * I / max(area * max(ct, 0.05), 1e-12) * colour
            else:
                r = max(L.get("radius", 0.0), dist * 1e-4)
                zc = (Vr @ rel)[2]
                A, B = np.array([fx * r / zc, 0.0]), np.array([0.0, fy * r / zc])
                round_face = True
                face = I / (r * r) * colour               # pi I / (pi r^2)
            blur = 0.0
            if focal_mm and n and focus_m:
                dm = max(dist * metres_per_unit, 0.05)
                blur = (focal_mm / 1000 / n) * abs(1 / dm - 1 / max(focus_m, 0.05)) * fx / diag
        uv = px / np.array([W, H])
        if not (-0.6 < uv[0] < 1.6 and -0.6 < uv[1] < 1.6):
            continue
        # A light leaving the picture still reaches the front glass a while: its ghosts
        # and glare fade out from 0.2 to 0.6 frame widths beyond the edge, rather than
        # stopping at a line (the face itself just slides out of the frame).
        beyond = max(-uv[0], uv[0] - 1, -uv[1], uv[1] - 1)
        fade = 1 - _smoothstep(0.2, 0.6, beyond)
        if fade <= 0:
            continue
        e = e * fade
        size = max(math.hypot(*A), math.hypot(*B)) / diag     # face size over the diagonal
        S = L.get("spectrum")
        if S is None:
            S = spectra.spectrum('COLOUR', L["colour"])
        out.append(dict(px=px, rgb=e * (math.pi / omega), dist=dist, blur=blur, A=A, B=B, radius=size,
                        round=round_face, face=np.asarray(face, np.float64), vis=1.0, name=L.get("name"),
                        spectrum=S, shares=spectra.band_shares(S)))
    return out


def visibility(srcs, alpha, depth):
    """How much of each light the frame shows: 16 samples over its face, seen where nothing
    in the frame is nearer than the light (alpha [H,W], depth = distance along the ray)."""
    H, W = alpha.shape
    for s in srcs:
        rf = max(2.0, math.sqrt(abs(s["A"][0] * s["B"][1] - s["A"][1] * s["B"][0])))
        seen = 0
        for k in range(16):
            r = math.sqrt((k + 0.5) / 16) * rf
            a = k * 2.39996323
            x, y = s["px"][0] + r * math.cos(a), s["px"][1] + r * math.sin(a)
            if x < 0 or y < 0 or x >= W or y >= H:
                seen += 1
                continue
            i, j = int(y), int(x)
            z = depth[i, j]
            if alpha[i, j] < 0.4 or not np.isfinite(z) or z > s["dist"] * 0.97:
                seen += 1
        s["vis"] = seen / 16
        # The face itself is hidden pixel by pixel behind whatever is nearer (render).
        s["cover"] = (alpha, depth)
    return srcs


# ------------------------------------------------------------------ the image
def _box(cx, cy, r, W, H):
    x0, x1 = int(max(0, math.floor(cx - r))), int(min(W, math.ceil(cx + r) + 1))
    y0, y1 = int(max(0, math.floor(cy - r))), int(min(H, math.ceil(cy + r) + 1))
    if x0 >= x1 or y0 >= y1:
        return None
    return slice(y0, y1), slice(x0, x1)


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _aperture(dx, dy, P, soft):
    """Inside the aperture's shape (radius 1): 1 inside, soft edge; blades, or round."""
    r = np.sqrt(dx * dx + dy * dy)
    nb, op = P["blades"], P["open"]
    if nb >= 3 and op < 0.98:
        seg = 2 * math.pi / nb
        th = np.arctan2(dy, dx) - P["rotation"]
        a = np.mod(th, seg) - seg / 2
        poly = r * np.cos(a) / math.cos(math.pi / nb)
        # wide open the blades are drawn back (round); stopped down, a polygon
        edge = poly + (r - poly) * _smoothstep(0.75, 0.98, op)
    else:
        edge = r
    m = 1 - _smoothstep(1 - soft, 1 + soft, edge)
    if P["obstruction"] > 0:                              # a mirror lens: a ring
        m = m * _smoothstep(P["obstruction"] - soft, P["obstruction"] + soft, r)
    return m


def glass_flaws(dx, dy, P, k):
    """The glass's own flaws as ghost k shows them (dx, dy: the point in the ghost, radius 1).
    Dust and seed bubbles cast small shadows (dust a dark speck with a faint bright fringe, a
    bubble a dim middle with a bright rim), cleaning marks and coating scratches catch the light
    as fine arcs and hairlines, fungus is a dark branching lace; the polish is never quite even.
    Vintage Click's glass_flaws, the same per lens and per ghost."""
    seed, dust = P["seed"], P["dust"]
    if dust <= 0 and not P["bubbles"]:
        return np.ones(np.broadcast(dx, dy).shape, np.float32)     # clean glass
    dx, dy = (np.array(a, np.float32) for a in np.broadcast_arrays(dx, dy))
    h = lambda x: _hash(seed, x)
    m = np.ones_like(dx)
    for i in range(int(dust * 4)):
        b = k * 37 + i * 3 + 200
        a, r = h(b) * 2 * math.pi, math.sqrt(h(b + 1)) * 0.92
        sz = 0.022 + (0.07 - 0.022) * h(b + 2)
        q = np.hypot(dx - r * math.cos(a), dy - r * math.sin(a)) / sz
        m = m * (1 - 0.55 * (1 - _smoothstep(0.7, 1.0, q)) + 0.18 * np.exp(-((q - 1.15) / 0.18) ** 2))
    if P["bubbles"]:
        for i in range(3):
            b = k * 37 + i * 3 + 500
            a, r = h(b) * 2 * math.pi, math.sqrt(h(b + 1)) * 0.85
            sz = 0.035 + (0.10 - 0.035) * h(b + 2)
            q = np.hypot(dx - r * math.cos(a), dy - r * math.sin(a)) / sz
            m = m * (1 - 0.35 * (1 - _smoothstep(0.6, 0.95, q)) + 0.45 * np.exp(-((q - 1.0) / 0.12) ** 2))
    marks = np.zeros_like(dx)
    for j in range(4):
        b = k * 37 + j * 4 + 800
        a = h(b) * 2 * math.pi
        cx, cy = 1.4 * math.cos(a), 1.4 * math.sin(a)
        R = 1.4 + (-0.8 + 1.4 * h(b + 1))
        arc = np.abs(np.hypot(dx - cx, dy - cy) - R)
        along = np.arctan2(dy - cy, dx - cx) - (a + math.pi)
        span = 0.25 + 0.5 * h(b + 2)
        wrap = np.abs(np.mod(along + 3 * math.pi, 2 * math.pi) - math.pi)
        marks += np.exp(-(arc / 0.006) ** 2) * (1 - _smoothstep(span * 0.7, span, wrap))
    scratches = np.zeros_like(dx)
    for j in range(int(dust * 2)):
        b = k * 37 + j * 5 + 1100
        a = h(b) * math.pi
        ox, oy = (h(b + 1) - 0.5) * 1.6, (h(b + 2) - 0.5) * 1.6
        along = (dx - ox) * math.cos(a) + (dy - oy) * math.sin(a)
        off = -(dx - ox) * math.sin(a) + (dy - oy) * math.cos(a)
        ln = 0.15 + 0.6 * h(b + 3)
        scratches += np.exp(-(off / 0.004) ** 2) * (1 - _smoothstep(ln * 0.8, ln, np.abs(along)))
    lace = np.zeros_like(dx)
    if P["fungus"]:
        for c in range(2):
            b = k * 37 + c * 9 + 1400
            a, r = h(b) * 2 * math.pi, math.sqrt(h(b + 1)) * 0.8
            qx, qy = dx - r * math.cos(a), dy - r * math.sin(a)
            size = 0.12 + 0.18 * h(b + 2)
            rr = np.hypot(qx, qy) / size
            th = np.arctan2(qy, qx)
            best = np.full_like(dx, 1e3)
            for br in range(7):
                tb = br / 7 * 2 * math.pi + h(b + 3 + br)
                wob = tb + 0.25 * np.sin(rr * 9 + br * 2.1) + 0.12 * np.sin(rr * 23 + br)
                dth = np.abs(np.mod(th - wob + 3 * math.pi, 2 * math.pi) - math.pi)
                best = np.minimum(best, dth * rr * size)
                tb2 = wob + 0.6 * (h(b + 20 + br) - 0.5) * 2
                dth2 = np.abs(np.mod(th - tb2 - 0.4 * np.sin(rr * 13) + 3 * math.pi, 2 * math.pi) - math.pi)
                best = np.where(rr > 0.35, np.minimum(best, dth2 * rr * size * 1.4), best)
            strand = np.exp(-(best / 0.006) ** 2) * (1 - _smoothstep(0.6, 1.0, rr))
            lace += np.where(rr < 1.0, strand + 0.6 * (1 - _smoothstep(0.0, 0.08, rr)), 0.0)
    polish = 0.05 * np.sin(dx * 6.1 + seed) * np.sin(dy * 4.7 + seed * 1.7)
    lc = np.minimum(lace, 1.0)
    return np.maximum(m * (1 + polish) * (1 - 0.6 * lc) + marks * 0.1 * dust + scratches * 0.3
                      + 0.15 * lc * (1 - lc) * 4, 0.0)


def render(srcs, P, W, H, axis):
    """The flare and the lights' faces, linear [H,W,3] float32, added to the frame's light."""
    add = np.zeros((H, W, 3), np.float32)
    if not srcs:
        return add
    diag = math.hypot(W, H)
    npx = W * H
    single = P["single"]
    seed = P["seed"]
    ax = np.asarray(axis, np.float64)
    ys = np.arange(H, dtype=np.float32) + 0.5
    xs = np.arange(W, dtype=np.float32) + 0.5
    for s in srcs:
        vis = s["vis"]
        if vis <= 0.001:
            continue
        I = (s["rgb"] * vis).astype(np.float32)
        sp = s["px"]
        v = sp - ax
        blur_px = s["blur"] * diag
        dX = xs[None, :] - np.float32(sp[0])
        dY = ys[:, None] - np.float32(sp[1])
        dg = np.sqrt(dX * dX + dY * dY)

        # Its lit face at its true brightness, spread by its own defocus; and the bloom.
        face = s["face"].astype(np.float32)
        if face.max() > 0:
            A, B = s["A"], s["B"]
            det = A[0] * B[1] - A[1] * B[0]
            rf = math.sqrt(max(abs(det), 0.25))
            grow = (rf + blur_px) / rf
            soft = (0.7 + blur_px) / (rf * grow) + 0.02
            reach = max(math.hypot(*A), math.hypot(*B), rf) * grow * (1 + soft) + 2
            box = _box(sp[0], sp[1], reach, W, H)
            if box is not None:
                bx = xs[box[1]][None, :] - sp[0]
                by = ys[box[0]][:, None] - sp[1]
                if abs(det) > 1e-6:
                    a1 = (bx * B[1] - by * B[0]) / det
                    a2 = (A[0] * by - A[1] * bx) / det
                else:
                    a1, a2 = bx / rf, by / rf
                rho = (np.sqrt(a1 * a1 + a2 * a2) if s["round"] else np.maximum(np.abs(a1), np.abs(a2))) / grow
                m = 1 - _smoothstep(1 - soft, 1 + soft, rho)
                cover = s.get("cover")
                if cover is not None:
                    # Seen through whatever is nearer than the light: a trunk in front of a
                    # lamp cuts into its face (the light's own pixels, not a dimmed whole).
                    al, dp = cover[0][box], cover[1][box]
                    nearer = np.isfinite(dp) & (dp < s["dist"] * 0.97)
                    m = m * np.where(nearer, 1 - np.clip(al, 0.0, 1.0), 1.0)
                else:
                    m = m * vis
                add[box] += (face / (grow * grow))[None, None, :] * m[..., None]
            lum = float(face @ LUMA)
            flux = lum * vis * math.pi * rf * rf
            r0 = 4.0 + 0.6 * rf + blur_px
            bloom = (0.05 if single else 0.025) / (math.pi * r0 * r0) / (1 + (dg / r0) ** 2) ** 2
            add += (face / max(lum, 1e-6) * flux)[None, None, :] * bloom[..., None]

        # Veiling glare: a broad haze, strongest near the light.
        rg = diag * 0.35
        # worn glass (dust, scratches, fungus, haze) scatters more of the light into the haze
        wear = 1 + 0.35 * P["dust"] + (0.8 if P["fungus"] else 0.0)
        glare = (0.006 if single else 0.0025) * wear / (1 + (dg / rg) ** 2)
        add += I[None, None, :] * (glare * (npx / (math.pi * rg * rg * 3.0)))[..., None]

        # Ghosts along the line through the axis, each the aperture's shape.
        for k in range(7 if single else 6):
            t = -1.6 + 3.1 * _hash(seed, k)
            rad = diag * (0.015 + 0.095 * _hash(seed, k + 20)) * (0.55 + 0.45 * P["open"])
            g = (2e-4 if single else 2e-5) * (0.4 + 1.2 * _hash(seed, k + 40))
            hue = _hash(seed, k + 60)
            tint = (np.array([1.0, 0.55, 0.2]) * (1 - hue) + np.array([0.3, 0.9, 0.5]) * hue) if single else \
                (np.array([0.45, 0.85, 0.55]) * (1 - hue) + np.array([0.85, 0.45, 0.9]) * hue)
            tint = tint * 0.75 + 0.25
            cg = ax + v * t
            field = math.hypot(*(cg - ax)) / diag
            focus_ghost = blur_px / rad * (0.15 + 1.05 * _hash(seed, k + 90))
            soft = 0.05 + 0.08 * _hash(seed, k + 80) + P["field"] * field + focus_ghost
            # wavelength by wavelength: bands of the light's spectrum, each its share of the
            # light's colour, bent by the dispersion and smeared over its own span
            bend_k = P["dispersion"] * (2.0 if single else 1.0) * (1 + 2 * abs(t))
            bends, widths = spectra.band_bend()
            spread = 1 + math.hypot(*(v * t)) / rad
            scs = [1 + bend_k * b for b in bends]
            reach = rad * max(scs) * (1 + soft + 0.6 * bend_k * spread) * max(P["squeeze"], 1.0) \
                + abs(t) * math.hypot(*v) * bend_k * 1.6 + 2
            box = _box(cg[0], cg[1], reach, W, H)
            if box is None:
                continue
            bx = xs[box[1]][None, :]
            by = ys[box[0]][:, None]
            m3 = np.zeros((by.shape[0], bx.shape[1], 3), np.float32)
            for bd, sc in enumerate(scs):
                c = ax + v * t * sc
                dx = (bx - c[0]) / (rad * sc) / max(P["squeeze"], 0.5)   # an anamorphic stretches them
                dy = (by - c[1]) / (rad * sc)
                sb = soft + 0.5 * bend_k * widths[bd] * spread
                m3 += _aperture(dx, dy, P, sb)[..., None] * s["shares"][bd].astype(np.float32)
            if m3.max() <= 0:
                continue
            m3 *= glass_flaws((bx - cg[0]) / rad / max(P["squeeze"], 0.5), (by - cg[1]) / rad, P, k)[..., None]
            rr = np.sqrt((bx - cg[0]) ** 2 + (by - cg[1]) ** 2) / rad
            rim = 0.8 + 0.4 * _smoothstep(0.6, 1.0, rr)
            add[box] += (I * tint.astype(np.float32))[None, None, :] * m3 * \
                (g * rim * npx / (math.pi * rad * rad))[..., None]

        # Scratches in the front coating: each throws a faint soft streak through the light.
        for j in range(min(3, int(P["dust"]))):
            a = _hash(seed, 1600 + j) * math.pi
            c, sn = math.cos(a), math.sin(a)
            along = np.abs(dX * c + dY * sn)
            across = np.abs(-dX * sn + dY * c)
            Ls = diag * (0.012 + 0.023 * _hash(seed, 1610 + j))
            wS = max(1.5, diag * 0.0012)
            streak = np.exp(-0.5 * (across / wS) ** 2) * np.exp(-(along / Ls) ** 1.5)
            add += I[None, None, :] * (1e-5 * streak * npx / (Ls * 2 * wS * 2.5))[..., None].astype(np.float32)
        # A faint ring round the light (a reflection off the film and back).
        rr0, w = diag * 0.18, diag * 0.012
        ring = np.zeros((H, W, 3), np.float32)
        ring_w = w * (1 + P["field"] * 4)
        near = np.abs(dg - rr0) < 5 * ring_w + rr0 * P["dispersion"] * 3 * 1.6
        if near.any():
            for bd, b in enumerate(spectra.band_bend()[0]):
                rc = rr0 * (1 + P["dispersion"] * 3 * b)
                ring += np.where(near, np.exp(-0.5 * ((dg - rc) / ring_w) ** 2), 0.0)[..., None] \
                    * s["shares"][bd].astype(np.float32)
        add += I[None, None, :] * ring * ((1e-4 if single else 2e-5) * npx / (2 * math.pi * rr0 * w * 2.5))

        # A filter on the lens: its flat glass and the front element reflect the light back,
        # a small sharp ghost mirrored through the axis.
        if P.get("glass"):
            cm = 2 * ax - sp
            rm = max(s.get("radius", 0.0) * diag * 1.3, 2.5)
            box = _box(cm[0], cm[1], rm * 4, W, H)
            if box is not None:
                dm2 = (xs[box[1]][None, :] - cm[0]) ** 2 + (ys[box[0]][:, None] - cm[1]) ** 2
                mg = np.exp(-0.5 * dm2 / (rm * rm))
                add[box] += I[None, None, :] * (8e-4 * mg * npx / (2 * math.pi * rm * rm))[..., None]
        # A cross screen or grating: the filter's own spikes from this light.
        spn = P.get("star_points", 0)
        if spn > 0:
            lines = max(1, spn // 2)
            Ls = diag * (0.22 if P["spectral"] else 0.16)
            star = np.zeros((H, W, 3), np.float32)
            for j in range(lines):
                ang = 0.35 + j * math.pi / lines
                c, sn = math.cos(ang), math.sin(ang)
                along = np.abs(dX * c + dY * sn)
                across = np.abs(-dX * sn + dY * c)
                spk = np.exp(-0.5 * (across / 1.4) ** 2) * np.exp(-3.2 * along / Ls)
                on = spk > 1e-5                     # only where the spike reaches
                if P["spectral"]:
                    # the light's own spectrum along the spike, order by order
                    u = along[on] / Ls
                    m = np.clip((u - 0.02) / 0.08, 0, 1)
                    m = m * m * (3 - 2 * m)
                    col = I[None, :] * (1 - m)[:, None] + spectra.grating_lamp(u, s["spectrum"], I) / 1.3 * m[:, None]
                    star[on] += (col * spk[on][:, None]).astype(np.float32)
                else:
                    star[on] += I[None, :] * spk[on][:, None]
            star *= (dg > 0.5)[..., None]
            add += star * np.float32(P["star_strength"] * 0.01 * npx / (lines * Ls * 2 * 1.4 * 2.5))

        # Starburst: diffraction at the corners of the iris. Wide open the blades are drawn
        # back to a circle (no corners); stopped down their corners throw spikes, longer the
        # smaller the opening. Only a light in focus is sharp enough to show them.
        nb = P["blades"]
        corners = 1 - float(_smoothstep(0.75, 0.98, P["open"]))
        sharp = math.exp(-blur_px / 3.0)
        if nb >= 3 and corners * sharp > 0.001:
            rays = nb if nb % 2 == 0 else 2 * nb
            th = np.arctan2(dY, dX) - P["rotation"]
            seg = 2 * math.pi / rays
            off = np.mod(th, seg) - seg / 2
            wpx = 1.1 + 0.5 * np.exp(-dg / 40.0)
            spike = np.exp(-0.5 * (off * dg / wpx) ** 2)
            length = diag * min(0.3, 0.025 / max(P["open"], 0.05))
            fade = (length / (length + dg)) ** 2 * np.exp(-dg / (length * 3.0))
            star = 6e-5 * corners * sharp * spike * fade * npx / (rays * length * 2.0 * wpx * 2.5 + 1.0)
            star = np.where(dg > 0.5, star, 0.0)
            add += I[None, None, :] * star[..., None].astype(np.float32)
    return add
