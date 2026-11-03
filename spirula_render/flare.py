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

Brightness is physical, in the relighting's units: a white surface lit by a light renders as
that light's relative irradiance there, so the flare is what that light sends into the lens,
spread as the lens spreads a small part of it. A light hidden behind splats or meshes (seen
through the depth of the finished frame) flares only as much as it shows.
"""

import math

import numpy as np

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


def lens_params(lens, n):
    return dict(blades=int(lens["blades"]), rotation=float(lens["bladeRotation"]),
                open=min(1.0, float(lens["maxAperture"]) / n), single=single_coated(lens),
                seed=_seed(lens["name"]), obstruction=float(lens.get("obstruction", 0.0)),
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
        out.append(dict(px=px, rgb=e * (math.pi / omega), dist=dist, blur=blur, A=A, B=B,
                        round=round_face, face=np.asarray(face, np.float64), vis=1.0, name=L.get("name")))
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
                add[box] += (face * vis / (grow * grow))[None, None, :] * m[..., None]
            lum = float(face @ LUMA)
            flux = lum * vis * math.pi * rf * rf
            r0 = 4.0 + 0.6 * rf + blur_px
            bloom = (0.05 if single else 0.025) / (math.pi * r0 * r0) / (1 + (dg / r0) ** 2) ** 2
            add += (face / max(lum, 1e-6) * flux)[None, None, :] * bloom[..., None]

        # Veiling glare: a broad haze, strongest near the light.
        rg = diag * 0.35
        glare = (0.006 if single else 0.0025) / (1 + (dg / rg) ** 2)
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
            scs = [1 + P["dispersion"] * (ch - 1) * (2.0 if single else 1.0) * (1 + 2 * abs(t)) for ch in range(3)]
            reach = rad * max(scs) * (1 + soft) * max(P["squeeze"], 1.0) + abs(t) * diag * P["dispersion"] * 3 + 2
            box = _box(cg[0], cg[1], reach, W, H)
            if box is None:
                continue
            bx = xs[box[1]][None, :]
            by = ys[box[0]][:, None]
            m3 = np.empty((by.shape[0], bx.shape[1], 3), np.float32)
            for ch, sc in enumerate(scs):
                c = ax + v * t * sc
                dx = (bx - c[0]) / (rad * sc) / max(P["squeeze"], 0.5)   # an anamorphic stretches them
                dy = (by - c[1]) / (rad * sc)
                m3[..., ch] = _aperture(dx, dy, P, soft)
            if m3.max() <= 0:
                continue
            rr = np.sqrt((bx - cg[0]) ** 2 + (by - cg[1]) ** 2) / rad
            rim = 0.8 + 0.4 * _smoothstep(0.6, 1.0, rr)
            add[box] += (I * tint.astype(np.float32))[None, None, :] * m3 * \
                (g * rim * npx / (math.pi * rad * rad))[..., None]

        # A faint ring round the light (a reflection off the film and back).
        rr0, w = diag * 0.18, diag * 0.012
        ring = np.empty((H, W, 3), np.float32)
        for ch in range(3):
            rc = rr0 * (1 + P["dispersion"] * 3 * (ch - 1))
            ring[..., ch] = np.exp(-0.5 * ((dg - rc) / (w * (1 + P["field"] * 4))) ** 2)
        add += I[None, None, :] * ring * ((1e-4 if single else 2e-5) * npx / (2 * math.pi * rr0 * w * 2.5))

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
