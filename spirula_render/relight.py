"""Relighting 3DGS splats with Blender lights, including self-shadowing.

The scan's colours already contain the lighting it was captured under. With
even, diffuse capture lighting they are close to albedo, so a splat lit by the
new lights gets

    colour = albedo * (ambient + exposure * sum_l  light_l / pi * max(n.l, 0) * visible_l)

computed in linear light (the model's colours are sRGB). Ambient 1 with no
lights gives back the scan as captured. The whole SH expansion of each splat is
scaled by the same factor, so sheen and view-dependent colour dim and brighten
with the light.

n is the splat's thinnest axis, turned towards the camera. visible_l is the
light left after passing through every splat between the light and this one
(a deep opacity shadow map), so fur shadows gradually rather than as a hard
surface.
"""

import math

import numpy as np

from . import native

C0 = 0.28209479177387814


def srgb_to_linear(c):
    c = np.maximum(c, 0.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.maximum(c, 0.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def thin_axes(quats, log_scales):
    """Each splat's thinnest axis (unit), object space."""
    q = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    cols = (
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], 1),
        np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], 1),
        np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1),
    )
    k = np.argmin(log_scales, axis=1)[:, None]
    n = np.where(k == 0, cols[0], np.where(k == 1, cols[1], cols[2]))
    return n.astype(np.float32)


def look_at(eye, target, up_hint=(0.0, 0.0, 1.0)):
    """World-to-camera, OpenCV axes (x right, y down, z forward)."""
    eye, target = np.asarray(eye, np.float64), np.asarray(target, np.float64)
    z = target - eye
    z /= np.linalg.norm(z)
    up = np.asarray(up_hint, np.float64)
    if abs(np.dot(up, z)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    V = np.eye(4)
    V[:3, :3] = R
    V[:3, 3] = -R @ eye
    return V


class DeepShadow:
    """Light transmittance at every depth, seen from one light (a deep opacity map).

    Fur is semi-transparent: light fades gradually as it passes through hair, so a
    splat's visibility is the product of (1 - opacity) of everything between it and
    the light, not a single "is it behind the surface" test. Each splat is binned
    into the light's pixel grid, each pixel's splats are sorted by distance, and a
    running sum of log transmittance gives the light left at any depth there.
    """

    def __init__(self, eye, target, corners, means, opacity, size, res):
        self.V = look_at(eye, target)
        pc = corners @ self.V[:3, :3].T + self.V[:3, 3]
        if np.any(pc[:, 2] <= 1e-9):
            raise ValueError("light is inside the model's bounds")
        t = np.max(np.abs(pc[:, :2]) / pc[:, 2:3])
        self.f = (res / 2) / (t * 1.05)
        self.res = res
        u, v, dist, z = self._project(means)
        iu = np.clip(np.floor(u).astype(np.int64), 0, res - 1)
        iv = np.clip(np.floor(v).astype(np.int64), 0, res - 1)
        # Share of a pixel each splat covers (its 2-sigma disc), so small splats
        # occlude a little and big ones fully.
        r_pix = 2.0 * size * self.f / np.maximum(z, 1e-9)
        cover = np.minimum(1.0, math.pi * r_pix * r_pix)
        a = np.clip(opacity * cover, 0.0, 0.999)
        self.dmax = float(dist.max()) * 1.0001 + 1e-9
        key = (iv * res + iu).astype(np.float64) + dist / self.dmax
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.cum = np.concatenate([[0.0], np.cumsum(np.log1p(-a[order]))])

    def _project(self, pts):
        pc = pts @ self.V[:3, :3].T.astype(np.float32) + self.V[:3, 3].astype(np.float32)
        z = np.maximum(pc[:, 2], 1e-9)
        u = self.f * pc[:, 0] / z + self.res / 2
        v = self.f * pc[:, 1] / z + self.res / 2
        return u, v, np.linalg.norm(pc, axis=1), z

    def visibility(self, means, bias, radius=1):
        """Light reaching each splat centre, 0..1, averaged over nearby pixels for soft edges."""
        u, v, dist, _ = self._project(means)
        d = np.maximum(dist - bias, 0.0) / self.dmax
        total = np.zeros(len(means), np.float64)
        taps = 0
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                iu = np.clip(np.floor(u).astype(np.int64) + du, 0, self.res - 1)
                iv = np.clip(np.floor(v).astype(np.int64) + dv, 0, self.res - 1)
                pix = (iv * self.res + iu).astype(np.float64)
                start = np.searchsorted(self.key, pix, side="left")
                stop = np.searchsorted(self.key, pix + d, side="left")
                total += np.exp(self.cum[stop] - self.cum[start])
                taps += 1
        return (total / taps).astype(np.float32)


class Relighter:
    """Holds the scan's original colours and turns lights into new ones."""

    def __init__(self, splats):
        self.means = splats["means"]
        self.dc0 = splats["dc"].astype(np.float32)
        self.sh0 = splats["sh"]
        self.normals = thin_axes(splats["quats"], splats["log_scales"])
        size = np.exp(np.max(splats["log_scales"], axis=1))
        self.splat_size = size.astype(np.float32)
        ls = np.sort(splats["log_scales"], axis=1)
        self.occluder_size = np.exp((ls[:, 1] + ls[:, 2]) / 2).astype(np.float32)   # face-on radius
        op = 1 / (1 + np.exp(-splats["logit_opacities"]))
        self.opacity = op.astype(np.float32)
        solid = self.means[op > 0.3] if np.count_nonzero(op > 0.3) > 100 else self.means
        lo, hi = np.percentile(solid, 0.5, axis=0), np.percentile(solid, 99.5, axis=0)
        self.centre = (lo + hi) / 2
        self.extent = float(np.linalg.norm(hi - lo))
        self.corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                 for z in (lo[2], hi[2])], np.float64)
        c = 0.5 + C0 * self.dc0
        self.c = c
        self.albedo_lin = srgb_to_linear(c)          # captured colour as albedo

    def colours(self, lights, cam_pos, ambient, exposure, shadows, shadow_res, softness,
                primitive, report=print):
        """lights: dicts in object space (type, pos, dir, colour rgb, power W or strength,
        spot cos limits, world_per_object). Returns new dc [N,3] and sh [N,K,3]."""
        n = self.normals.copy()
        to_cam = cam_pos[None, :] - self.means
        flip = np.einsum('ij,ij->i', n, to_cam) < 0
        n[flip] *= -1
        E = np.full((len(n), 3), ambient, np.float32)
        gain = 2.0 ** exposure
        bias = 2.0 * self.splat_size                 # don't let a splat shadow itself
        for L in lights:
            if L["type"] == 'SUN':
                ldir = -L["dir"] / np.linalg.norm(L["dir"])     # towards the light
                irr = np.full(len(n), L["strength"], np.float32)
                cosv = n @ ldir.astype(np.float32)
                eye = self.centre + ldir * self.extent * 50
            else:
                d = L["pos"][None, :] - self.means
                dist = np.linalg.norm(d, axis=1)
                ldirs = d / np.maximum(dist[:, None], 1e-12)
                cosv = np.einsum('ij,ij->i', n, ldirs)
                dist_w = dist * L["world_per_object"]
                irr = L["power"] / (4 * math.pi * np.maximum(dist_w, 1e-6) ** 2)
                if L["type"] == 'SPOT':
                    c = -(ldirs @ (L["dir"] / np.linalg.norm(L["dir"])).astype(np.float32))
                    cos_out, cos_in = L["spot_cos"]
                    irr = irr * np.clip((c - cos_out) / max(cos_in - cos_out, 1e-6), 0, 1)
                eye = L["pos"]
            vis = 1.0
            if shadows:
                try:
                    sm = DeepShadow(eye, self.centre, self.corners, self.means, self.opacity,
                                    self.occluder_size, shadow_res)
                    vis = sm.visibility(self.means, bias, softness)
                except ValueError as e:
                    report(f"Spirula: no shadows from {L['name']}: {e}")
            E += (gain * irr / math.pi * np.maximum(cosv, 0) * vis)[:, None] * \
                np.asarray(L["colour"], np.float32)[None, :]
        lit = linear_to_srgb(self.albedo_lin * E)
        # Ratio of new to captured colour; splats captured black stay as they were.
        f = np.where(self.c > 1e-4, lit / np.maximum(self.c, 1e-4), 1.0).astype(np.float32)
        dc = np.where(self.c > 1e-4, (lit - 0.5) / C0, self.dc0).astype(np.float32)
        sh = None if self.sh0 is None else (self.sh0 * f[:, None, :]).astype(np.float32)
        return dc, sh
