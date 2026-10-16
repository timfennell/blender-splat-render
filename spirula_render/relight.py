"""Relighting 3DGS splats with Blender lights, including self-shadowing.

Made for scans captured with the light fixed to the camera (the subject turned
between shots): every photo was lit from where it was taken, so each splat's SH
records how it looks with light and view in the same direction. Relighting reads
that record directly (see Relighter.colours): the broad part (DC + band 1) in
each new light's direction gives the diffuse response, and bands 2-3 at the half
vector between light and camera give the shine. A light at the camera gives the
scan back; Ambient 1 with no lights gives it back exactly.

Shadows come from a deep opacity map per light: the light left after passing
through every splat between the light and this one, so fur shadows gradually.
"""

import math

import numpy as np

from . import native

C0 = 0.28209479177387814
C1 = 0.4886025119029199
_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
       -1.0925484305920792, 0.5462742152960396)
_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
       -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def sh_basis(d, K):
    """3DGS real SH basis at unit directions d [M,3]: [M, 1+K] (DC first)."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    cols = [np.full_like(x, C0), -C1 * y, C1 * z, -C1 * x]
    c = _C2
    cols += [c[0] * x * y, c[1] * y * z, c[2] * (2 * zz - xx - yy), c[3] * x * z, c[4] * (xx - yy)]
    c = _C3
    cols += [c[0] * y * (3 * xx - yy), c[1] * x * y * z, c[2] * y * (4 * zz - xx - yy),
             c[3] * z * (2 * zz - 3 * xx - 3 * yy), c[4] * x * (4 * zz - xx - yy),
             c[5] * z * (xx - yy), c[6] * x * (xx - 3 * yy)]
    return np.stack(cols[:1 + K], 1)


def srgb_to_linear(c):
    c = np.maximum(c, 0.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.maximum(c, 0.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


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
        self.c = c.astype(np.float32)
        self.b1 = (np.ascontiguousarray(self.sh0[:, :3], np.float32)
                   if self.sh0 is not None and self.sh0.shape[1] >= 3 else None)
        self.albedo_lin = srgb_to_linear(c)          # captured colour as albedo

    # ---------------------------------------------------------- light model
    def _light_terms(self, L, gain):
        """Per-splat direction towards the light (unit), relative irradiance e (1 = the
        capture light), the shadow-map eye, and the light's mean direction."""
        if L["type"] == 'SUN':
            ldir = (-L["dir"] / np.linalg.norm(L["dir"])).astype(np.float32)
            e = np.full(len(self.means), gain * L["strength"], np.float32)
            return np.broadcast_to(ldir, self.means.shape), e, self.centre + ldir * self.extent * 50, ldir
        d = (L["pos"][None, :] - self.means).astype(np.float32)
        dist = np.sqrt(np.einsum('ij,ij->i', d, d))
        ldirs = d / np.maximum(dist[:, None], 1e-12)
        e = gain * L["power"] / (4 * math.pi * np.maximum(dist * L["world_per_object"], 1e-6) ** 2)
        if L["type"] == 'SPOT':
            c = -(ldirs @ (L["dir"] / np.linalg.norm(L["dir"])).astype(np.float32))
            cos_out, cos_in = L["spot_cos"]
            e = e * np.clip((c - cos_out) / max(cos_in - cos_out, 1e-6), 0, 1)
        mean_dir = L["pos"] - self.centre
        mean_dir = (mean_dir / max(np.linalg.norm(mean_dir), 1e-12)).astype(np.float32)
        return ldirs, e.astype(np.float32), L["pos"], mean_dir

    def diffuse_at(self, dirs):
        """The captured colour's broad part (DC + band 1) at SH directions `dirs` [N,3]
        (camera->splat, as 3DGS reads them), clamped at 0.

        The capture light sat at the camera, so this is how the splat looks lit from that
        direction: for a matte surface, its diffuse response to a light from there."""
        if self.b1 is None:
            return np.maximum(self.c, 0.0)
        if dirs.strides[0] == 0:                     # one direction for all splats (a sun)
            x, y, z = (float(t) for t in dirs[0])
            c = self.c + np.einsum('nkc,k->nc', self.b1, np.array([-C1 * y, C1 * z, -C1 * x], np.float32))
        else:
            basis = C1 * np.stack([-dirs[:, 1], dirs[:, 2], -dirs[:, 0]], 1).astype(np.float32)
            c = self.c + np.einsum('nkc,nk->nc', self.b1, basis)
        return np.maximum(c, 0.0)

    def _shadow(self, eye, shadows, shadow_res, softness, report, name):
        if not shadows:
            return 1.0
        try:
            sm = DeepShadow(eye, self.centre, self.corners, self.means, self.opacity,
                            self.occluder_size, shadow_res)
            return sm.visibility(self.means, 2.0 * self.splat_size, softness)
        except ValueError as e:
            report(f"Spirula: no shadows from {name}: {e}")
            return 1.0

    def quick_dc(self, lights, cam_pos, ambient, exposure):
        """Fast base-colour relight for dragging lights: diffuse only, no shadows, no SH,
        mixed in display space (an approximation the full pass replaces)."""
        out = self.c * np.float32(ambient ** (1 / 2.2))
        gain = 2.0 ** exposure
        for L in lights:
            ldirs, e, _, _ = self._light_terms(L, gain)
            w = np.power(e, np.float32(1 / 2.2))[:, None] * \
                np.asarray(L["colour"], np.float32)[None, :]
            out = out + self.diffuse_at(-ldirs) * w
        return ((out - 0.5) / C0).astype(np.float32)

    def sh_at(self, dirs, first, last):
        """Sum of SH coefficients first..last-1 (indices into the higher bands) at dirs [N,3]."""
        B = sh_basis(dirs, self.sh0.shape[1])[:, 1 + first:1 + last]          # [N, k]
        return np.einsum('nk,nkc->nc', B, self.sh0[:, first:last])

    def colours(self, lights, cam_pos, ambient, exposure, shadows, shadow_res, softness,
                primitive, report=print, diffuse_only=False):
        """One relit colour per splat for this camera (the renderer evaluates SH once per
        splat, from the direction to the camera, so that's all it needs). Returns new dc
        [N,3] and None: render with the higher SH bands switched off.

        diffuse : the captured broad colour (DC + band 1) read in each light's direction --
                  the capture light sat at the camera, so that is how the splat looks lit
                  from there -- times the light's relative irradiance and shadow, summed in
                  linear light.
        shine   : SH bands 2-3 read at the half vector between the light and the camera.
                  The capture had light = view, so its half vector was the view itself.
        ambient : a share of the scan exactly as captured (all bands, seen from the camera).
        With one light at the camera and no ambient this gives back the scan."""
        gain = 2.0 ** exposure
        # 3DGS reads SH at the camera->splat direction.
        v = (self.means - cam_pos[None, :]).astype(np.float32)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        lin = self.albedo_lin * ambient
        shine = np.zeros_like(self.c)
        has_sh = self.sh0 is not None and self.sh0.shape[1] >= 8 and not diffuse_only
        if has_sh and ambient > 0:
            shine += ambient ** (1 / 2.2) * self.sh_at(v, 0, self.sh0.shape[1])
        for L in lights:
            ldirs, e, eye, _ = self._light_terms(L, gain)
            vis = self._shadow(eye, shadows, shadow_res, softness, report, L["name"])
            w = (e * vis)[:, None] * np.asarray(L["colour"], np.float32)[None, :]
            lin = lin + srgb_to_linear(self.diffuse_at(-ldirs)) * w
            if has_sh:
                h = v - ldirs
                h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
                shine += np.power(np.maximum(w, 0), 1 / 2.2) * self.sh_at(h, 3, self.sh0.shape[1])
        out = linear_to_srgb(lin) + shine
        return ((out - 0.5) / C0).astype(np.float32), None
