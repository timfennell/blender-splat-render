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

    def __init__(self, V, f, res, means, opacity, size):
        """V: world->light camera (OpenCV axes), f: focal length in pixels. Every splat
        given here must be in front of the light and inside its view."""
        self.V, self.f, self.res = V, f, res
        u, v, dist, z = self._project(means)
        iu = np.clip(np.floor(u).astype(np.int64), 0, res - 1)
        iv = np.clip(np.floor(v).astype(np.int64), 0, res - 1)
        # Share of a pixel each splat covers (its 2-sigma disc), so small splats
        # occlude a little and big ones fully.
        r_pix = 2.0 * size * self.f / np.maximum(z, 1e-9)
        cover = np.minimum(1.0, math.pi * r_pix * r_pix)
        a = np.clip(opacity * cover, 0.0, 0.999)
        self.dmax = (float(dist.max()) if len(dist) else 1.0) * 1.0001 + 1e-9
        pix = iv * res + iu
        key = pix.astype(np.float64) + dist / self.dmax
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.cum = np.concatenate([[0.0], np.cumsum(np.log1p(-a[order]))])
        # For each splat in this map: where its pixel's column starts and ends in the sorted
        # order, and its own place in it -- so what lies in front of it needs no search.
        n = len(order)
        sp = pix[order]
        new = np.ones(n, bool)
        new[1:] = sp[1:] != sp[:-1]
        start_sorted = np.maximum.accumulate(np.where(new, np.arange(n), 0))
        last = np.ones(n, bool)
        last[:-1] = sp[1:] != sp[:-1]
        end_sorted = np.minimum.accumulate(np.where(last, np.arange(n) + 1, n)[::-1])[::-1]
        self.rank = np.empty(n, np.int64)
        self.rank[order] = np.arange(n)
        self.col_start = start_sorted[self.rank]
        self.col_end = end_sorted[self.rank]
        self.pix_iu, self.pix_iv = iu, iv
        # Exact coverage per pixel from the renderer itself (every splat's full footprint),
        # seen from the light. The binned map above only knows each splat's centre pixel,
        # which undercounts coverage through a thin shell; it is used for depth ORDER only.
        self.gpu_T = None
        self.gpu_depth = None
        try:
            _, alpha, depth, _ = native.render_ex(np.asarray(V, np.float32), f, f, res / 2, res / 2,
                                                  res, res, "3dgs")
            self.gpu_T = np.clip(1.0 - alpha, 1e-6, 1.0).astype(np.float64)
            self.gpu_alpha, self.gpu_depth = alpha, depth
        except Exception:
            pass

    def _project(self, pts):
        pc = pts @ self.V[:3, :3].T.astype(np.float32) + self.V[:3, 3].astype(np.float32)
        z = np.maximum(pc[:, 2], 1e-9)
        u = self.f * pc[:, 0] / z + self.res / 2
        v = self.f * pc[:, 1] / z + self.res / 2
        return u, v, np.linalg.norm(pc, axis=1), z

    density = 1.0              # optical depth multiplier for shadows (thin shells block more)

    def self_visibility(self, softness=1, bias=0.0):
        """Light reaching each splat this map was built from (fast path: no searching).

        The renderer's exact coverage of the splat's column, raised to the share of that
        column's optical depth lying in front of the splat; softness blurs the coverage
        image over (2*softness+1)^2 pixels for soft shadow edges."""
        if np.any(np.asarray(bias) > 0):
            # only what lies more than `bias` in front counts (a splat's own surface layer
            # mustn't shade it): one search, inside the splat's own column
            q = self.key[self.rank] - np.asarray(bias, np.float64) / self.dmax
            q = np.maximum(q, np.floor(self.key[self.rank]))
            stop = np.minimum(np.searchsorted(self.key, q, side="left"), self.rank)
        else:
            stop = self.rank
        before = self.cum[stop] - self.cum[self.col_start]              # <= 0, self excluded
        whole = self.cum[self.col_end] - self.cum[self.col_start]
        frac = np.where(whole < -1e-12, before / np.minimum(whole, -1e-12), 0.0)
        if self.gpu_T is None:
            return np.exp(self.density * before).astype(np.float32)
        T = self.gpu_T
        if softness > 0:
            logT = np.log(T)
            k = 2 * softness + 1
            for axis in (0, 1):                                          # separable box blur
                c = np.cumsum(np.pad(logT, [(softness + 1, softness)] if axis == 0 else
                                     [(0, 0), (softness + 1, softness)], mode='edge'), axis=axis)
                logT = ((c[k:] - c[:-k]) if axis == 0 else (c[:, k:] - c[:, :-k])) / k
            T = np.exp(logT)
        t = T[self.pix_iv, self.pix_iu]
        return np.power(t, self.density * np.clip(frac, 0.0, 1.0)).astype(np.float32)

    def visibility(self, means, bias, radius=1):
        """Light reaching each splat centre, 0..1, averaged over nearby pixels for soft edges."""
        u, v, dist, _ = self._project(means)
        # points outside this light's view (mesh surfaces can be) have no splat before them
        pz = means @ self.V[2, :3].astype(np.float32) + np.float32(self.V[2, 3])
        seen = (pz > 1e-9) & (u >= 0) & (u < self.res) & (v >= 0) & (v < self.res)
        d = np.minimum(np.maximum(dist - bias, 0.0) / self.dmax, 0.9999)
        total = np.zeros(len(means), np.float64)
        taps = 0
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                iu = np.clip(np.floor(u).astype(np.int64) + du, 0, self.res - 1)
                iv = np.clip(np.floor(v).astype(np.int64) + dv, 0, self.res - 1)
                pix = (iv * self.res + iu).astype(np.float64)
                start = np.searchsorted(self.key, pix, side="left")
                stop = np.searchsorted(self.key, pix + d, side="left")
                if self.gpu_T is None:
                    total += np.exp(self.density * (self.cum[stop] - self.cum[start]))
                else:
                    end = np.searchsorted(self.key, pix + 1.0, side="left")
                    before = self.cum[stop] - self.cum[start]          # <= 0
                    whole = self.cum[end] - self.cum[start]            # <= 0
                    # share of this column's occlusion that lies in front of the splat
                    frac = np.where(whole < -1e-9, before / np.minimum(whole, -1e-9),
                                    (stop > start).astype(np.float64))
                    total += np.power(self.gpu_T[iv, iu], self.density * np.clip(frac, 0.0, 1.0))
                taps += 1
        return np.where(seen, total / taps, 1.0).astype(np.float32)


def _blurred_log_T(T, softness):
    logT = np.log(T)
    if softness > 0:
        k = 2 * softness + 1
        for axis in (0, 1):                                              # separable box blur
            c = np.cumsum(np.pad(logT, [(softness + 1, softness)] if axis == 0 else
                                 [(0, 0), (softness + 1, softness)], mode='edge'), axis=axis)
            logT = ((c[k:] - c[:-k]) if axis == 0 else (c[:, k:] - c[:, :-k])) / k
    return logT


def surface_visibility(sm, pts, bias, softness):
    """Light reaching points on other surfaces (meshes) through the splats of map `sm`.

    Like DeepShadow.visibility, but the coverage is read with bilinear filtering: a mesh
    surface is continuous, and nearest-pixel reads show the shadow map's pixels as steps
    where it meets the surface at a grazing angle."""
    u, v, dist, _ = sm._project(pts)
    pz = pts @ sm.V[2, :3].astype(np.float32) + np.float32(sm.V[2, 3])
    res = sm.res
    seen = (pz > 1e-9) & (u >= 0) & (u < res) & (v >= 0) & (v < res)
    out = np.ones(len(pts), np.float32)
    if sm.gpu_T is None or not np.any(seen):
        return np.where(seen, sm.visibility(pts, bias, softness), 1.0).astype(np.float32)
    u, v, dist = u[seen], v[seen], dist[seen]
    logT = _blurred_log_T(sm.gpu_T, softness)
    # share of each column's occlusion in front of the point (nearest column)
    iu = np.clip(np.floor(u).astype(np.int64), 0, res - 1)
    iv = np.clip(np.floor(v).astype(np.int64), 0, res - 1)
    pix = (iv * res + iu).astype(np.float64)
    d = np.minimum(np.maximum(dist - bias, 0.0) / sm.dmax, 0.9999)
    start = np.searchsorted(sm.key, pix, side="left")
    stop = np.searchsorted(sm.key, pix + d, side="left")
    end = np.searchsorted(sm.key, pix + 1.0, side="left")
    before = sm.cum[stop] - sm.cum[start]
    whole = sm.cum[end] - sm.cum[start]
    frac = np.where(whole < -1e-9, before / np.minimum(whole, -1e-9), 0.0)
    empty = whole >= -1e-9
    if np.any(empty) and sm.gpu_depth is not None:
        # A column with no splats: its coverage is only the neighbours' blurred edge. Whether
        # that edge is in front of the point: compare with the neighbours' mean depth.
        r = softness + 1
        k = 2 * r + 1
        a = sm.gpu_alpha.astype(np.float64)
        ad = a * sm.gpu_depth
        def box(img):
            for axis in (0, 1):
                c = np.cumsum(np.pad(img, [(r + 1, r)] if axis == 0 else [(0, 0), (r + 1, r)],
                                     mode='constant'), axis=axis)
                img = (c[k:] - c[:-k]) if axis == 0 else (c[:, k:] - c[:, :-k])
            return img
        D = box(ad) / np.maximum(box(a), 1e-9)
        de = dist[empty] - D[iv[empty], iu[empty]]
        frac[empty] = np.clip(de / max(4.0 * float(np.max(bias)) if np.ndim(bias) else 4.0 * bias, 1e-9) + 0.5,
                              0.0, 1.0)
    # coverage, bilinear between pixel centres
    x, y = u - 0.5, v - 0.5
    x0 = np.clip(np.floor(x).astype(np.int64), 0, res - 1)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, res - 1)
    x1, y1 = np.minimum(x0 + 1, res - 1), np.minimum(y0 + 1, res - 1)
    fx, fy = np.clip(x - x0, 0, 1), np.clip(y - y0, 0, 1)
    lt = (logT[y0, x0] * (1 - fx) * (1 - fy) + logT[y0, x1] * fx * (1 - fy) +
          logT[y1, x0] * (1 - fx) * fy + logT[y1, x1] * fx * fy)
    out[seen] = np.exp(sm.density * np.clip(frac, 0.0, 1.0) * lt)
    return out


def light_visibility(eye, target, corners, means, opacity, size, query, bias, res, softness,
                     density=1.0, tris=None, extra=None, extra_bias=0.0):
    """Light reaching each `query` splat from a light at `eye` (deep opacity shadows).

    One map aimed at the model when it fits in a reasonable field of view; otherwise
    (a light close to or inside the model) a cube of six maps around the light.
    tris: mesh triangles [T,3,3] that also block the light. extra: more points (mesh
    surfaces) to find the splats' shadow at; then returns (vis, vis_extra)."""
    from . import meshes
    eye = np.asarray(eye, np.float64)
    vis = np.ones(len(query), np.float32)
    vis_x = None if extra is None else np.ones(len(extra), np.float32)
    rel_all = face_all = face_q = face_x = None

    def faces(pts):
        rel = (pts - eye).astype(np.float32)
        ax = np.argmax(np.abs(rel), axis=1)
        return ax * 2 + (np.take_along_axis(rel, ax[:, None], 1)[:, 0] < 0)

    for V, f_unit, fi in _face_views(eye, target, corners):
        f = f_unit * res
        if fi is None:
            inside = slice(None)
            q = np.arange(len(query))
            xq = None if extra is None else np.arange(len(extra))
        else:
            if rel_all is None:
                rel_all = (means - eye).astype(np.float32)
                face_q = faces(query)
                face_x = None if extra is None else faces(extra)
            pcs = rel_all @ V[:3, :3].T.astype(np.float32)
            z = pcs[:, 2]
            # occluders: splats in front of this face, within its (slightly widened) view
            inside = (z > 1e-9) & (np.abs(pcs[:, 0]) <= 1.05 * z) & (np.abs(pcs[:, 1]) <= 1.05 * z)
            q = np.nonzero(face_q == fi)[0]
            xq = None if extra is None else np.nonzero(face_x == fi)[0]
            if len(q) == 0 and (xq is None or len(xq) == 0):
                continue
        sm = DeepShadow(V, f, res, means[inside], opacity[inside], size[inside])
        sm.density = density
        if query is means:
            if fi is None:
                vis[:] = sm.self_visibility(softness, bias)
            else:
                pos = np.full(len(means), -1, np.int64)
                pos[np.nonzero(inside)[0]] = np.arange(int(inside.sum()))
                pq = pos[q]
                ok = pq >= 0
                v = sm.self_visibility(softness, bias[inside] if np.ndim(bias) else bias)
                vis[q[ok]] = v[pq[ok]]
        elif len(q):
            vis[q] = sm.visibility(query[q], bias[q] if np.ndim(bias) else bias, softness)
        if tris is not None and len(q):
            dmap = meshes.rasterize_distance(V, f, res, tris)
            vis[q] *= meshes.mesh_visibility(dmap, V, f, res, query[q],
                                             bias[q] if np.ndim(bias) else bias, softness)
        if extra is not None and len(xq):
            vis_x[xq] = surface_visibility(sm, extra[xq], extra_bias, softness)
    return vis if extra is None else (vis, vis_x)


def _face_views(eye, target, corners):
    """The light's view(s): one aimed at the model, or a cube of six around a close light.
    Yields (V, f_per_unit_res, face_index or None)."""
    V = look_at(eye, target)
    pc = corners @ V[:3, :3].T + V[:3, 3]
    if np.all(pc[:, 2] > 1e-9):
        t = float(np.max(np.abs(pc[:, :2]) / pc[:, 2:3]))
        if t < 1.6:
            yield V, 0.5 / (t * 1.05), None
            return
    ups = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 1, 0), (0, 1, 0)]
    for axis in range(3):
        for sign_i, sign in enumerate((1.0, -1.0)):
            d = np.zeros(3); d[axis] = sign
            yield look_at(eye, np.asarray(eye, np.float64) + d, ups[axis * 2 + sign_i]), 0.5 / 1.05, axis * 2 + sign_i


def quick_light_visibility(eye, target, corners, means, width, res, density=1.0, tris=None,
                           bias=0.0, extra=None):
    """Cheap level-of-detail shadows for lights being moved: one render per view from the
    light gives coverage and the depth where coverage reaches 50%. A splat well behind that
    surface gets the column's coverage, one in front gets full light, blended across the fur
    (2*bias). No sorting, so it costs one projection of the splats."""
    eye = np.asarray(eye, np.float64)
    vis = np.ones(len(means), np.float32)
    vis_x = None if extra is None else np.ones(len(extra), np.float32)
    rel = (means - eye).astype(np.float32)
    rel_x = None if extra is None else (extra - eye).astype(np.float32)
    face = face_x = None

    def faces(r):
        ax = np.argmax(np.abs(r), axis=1)
        return ax * 2 + (np.take_along_axis(r, ax[:, None], 1)[:, 0] < 0)

    def lookup(r, alpha, median, V, f, w, outside_lit):
        pc = r @ V[:3, :3].T.astype(np.float32)
        z = np.maximum(pc[:, 2], 1e-9)
        u = f * pc[:, 0] / z + res / 2
        v = f * pc[:, 1] / z + res / 2
        iu = np.clip(u.astype(np.int64), 0, res - 1)
        iv = np.clip(v.astype(np.int64), 0, res - 1)
        dist = np.linalg.norm(pc, axis=1)
        a = alpha[iv, iu]
        med = np.where(a > 0.5, median[iv, iu], np.inf)
        # light is 50% at the median depth by definition, fades from full in front of it,
        # and can't fall below what gets through the whole column
        T_d = np.exp(-math.log(2.0) * np.exp(np.clip((dist - med) / np.maximum(w, 1e-12), -30, 30)))
        T_floor = np.power(np.clip(1.0 - a, 1e-6, 1.0), density)
        out = np.maximum(T_d, T_floor)
        if outside_lit:
            out = np.where((pc[:, 2] > 1e-9) & (u >= 0) & (u < res) & (v >= 0) & (v < res), out, 1.0)
        return out

    for V, f_unit, fi in _face_views(eye, target, corners):
        f = f_unit * res
        if fi is not None:
            if face is None:
                face = faces(rel)
                face_x = None if extra is None else faces(rel_x)
            q = np.nonzero(face == fi)[0]
            xq = None if extra is None else np.nonzero(face_x == fi)[0]
        else:
            q = np.arange(len(means))
            xq = None if extra is None else np.arange(len(extra))
        if len(q) == 0 and (xq is None or len(xq) == 0):
            continue
        _, alpha, _, median = native.render_ex(np.asarray(V, np.float32), f, f, res / 2, res / 2,
                                               res, res, "3dgs")
        w = np.asarray(width)[q] if np.ndim(width) else width
        vis[q] = lookup(rel[q], alpha, median, V, f, w, False)
        if extra is not None and len(xq):
            vis_x[xq] = lookup(rel_x[xq], alpha, median, V, f,
                               float(np.median(width)) if np.ndim(width) else width, True)
        if tris is not None:
            from . import meshes
            dmap = meshes.rasterize_distance(V, f, res, tris)
            vis[q] *= meshes.mesh_visibility(dmap, V, f, res, means[q],
                                             bias[q] if np.ndim(bias) else bias, 0)
    return vis if extra is None else (vis, vis_x)


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
            if "portal" in L:                     # light bounced off a mirror
                from . import meshes
                e = e * meshes.portal_mask(L["portal"], self.means, L)
            return np.broadcast_to(ldir, self.means.shape), e, self.centre + ldir * self.extent * 50, ldir
        d = (L["pos"][None, :] - self.means).astype(np.float32)
        dist = np.sqrt(np.einsum('ij,ij->i', d, d))
        ldirs = d / np.maximum(dist[:, None], 1e-12)
        e = gain * L["power"] / (4 * math.pi * np.maximum(dist * L["world_per_object"], 1e-6) ** 2)
        if L["type"] == 'SPOT':
            c = -(ldirs @ (L["dir"] / np.linalg.norm(L["dir"])).astype(np.float32))
            cos_out, cos_in = L["spot_cos"]
            e = e * np.clip((c - cos_out) / max(cos_in - cos_out, 1e-6), 0, 1)
        if "portal" in L:                         # light bounced off a mirror
            from . import meshes
            e = e * meshes.portal_mask(L["portal"], self.means, L)
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

    mesh_tris = None           # world triangles that also cast shadows (set by the engine)
    mesh_points = None         # mesh surface points to find the splats' shadow at
    mesh_vis = None            # -> light name: light reaching each mesh point (after colours())

    def _tris_for(self, L):
        """Mesh triangles that can block light L. Light bounced off a mirror comes from the
        light's image behind it: only what's in front of the mirror (the room side) counts,
        not the mirror itself."""
        tris = self.mesh_tris
        portal = L.get("portal")
        if tris is None or portal is None:
            return tris
        s = (tris @ np.asarray(portal["n"], np.float32) - np.float32(portal["off"])) * \
            np.float32(portal["side"])
        keep = np.all(s > 1e-5 * max(self.extent, 1e-3), axis=1)
        return tris[keep] if keep.any() else None

    def _shadow(self, eye, shadows, shadow_res, softness, report, name, L=None):
        if not shadows:
            if self.mesh_points is not None:
                self.mesh_vis[name] = np.ones(len(self.mesh_points), np.float32)
            return 1.0
        out = light_visibility(eye, self.centre, self.corners, self.means, self.opacity,
                               self.occluder_size, self.means, 2.0 * self.splat_size,
                               shadow_res, softness, self.shadow_density,
                               self._tris_for(L) if L is not None else self.mesh_tris,
                               self.mesh_points, 0.004 * self.extent)
        if self.mesh_points is None:
            return out
        self.mesh_vis[name] = out[1]
        return out[0]

    def quick_dc(self, lights, cam_pos, ambient, exposure, shadows=False, shadow_res=512):
        """Fast base-colour relight for dragging lights: diffuse only, no shadows, no SH,
        mixed in display space (an approximation the full pass replaces)."""
        out = self.c * np.float32(ambient ** (1 / 2.2))
        gain = 2.0 ** exposure
        self.mesh_vis = {}
        for L in lights:
            ldirs, e, eye, _ = self._light_terms(L, gain)
            if shadows:
                v = quick_light_visibility(eye, self.centre, self.corners, self.means,
                                           self.quick_width * self.extent, shadow_res,
                                           self.shadow_density, self._tris_for(L),
                                           2.0 * self.splat_size, self.mesh_points)
                if self.mesh_points is not None:
                    v, self.mesh_vis[L["name"]] = v
                e = e * v
            w = np.power(e, np.float32(1 / 2.2))[:, None] * \
                np.asarray(L["colour"], np.float32)[None, :]
            out = out + self.diffuse_at(-ldirs) * w
        return ((out - 0.5) / C0).astype(np.float32)

    shine_colour = 0.25        # 0: neutral (brightness only) ... 1: the full captured colour
    shadow_density = 4.0       # see SpirulaSettings.shadow_density
    quick_width = 0.08         # light falloff depth behind the lit surface (quick shadows), x extent

    def shine_at(self, h):
        """The shine seen at half vector h: brightness from SH bands 2-3, colour from the full
        captured colour there. Bands 2-3 also hold terms that only cancel the lower bands
        (often strongly coloured); evaluated apart from them those would show as false
        colour, so only their brightness is used."""
        K = self.sh0.shape[1]
        S = self.sh_at(h, 3, K)
        full = np.maximum(self.diffuse_at(h) + S, 0.0)
        lum_w = np.array([0.2126, 0.7152, 0.0722], np.float32)
        Ls = S @ lum_w
        Lf = full @ lum_w
        chroma = np.where(Lf[:, None] > 1e-3, full / np.maximum(Lf, 1e-3)[:, None], 1.0)
        k = np.float32(self.shine_colour)
        return Ls[:, None] * ((1 - k) + k * chroma)

    last_vis = {}

    def diffuse_dc(self, lights, ambient, exposure):
        """The last colours() relight without its view-dependent parts (shine, captured view
        colour), reusing its shadows: for views from elsewhere, such as reflections."""
        gain = 2.0 ** exposure
        lin = self.albedo_lin * ambient
        for L in lights:
            ldirs, e, _, _ = self._light_terms(L, gain)
            vis = np.broadcast_to(np.asarray(self.last_vis.get(L["name"], 1.0), np.float32), e.shape)
            w = (e * vis)[:, None] * np.asarray(L["colour"], np.float32)[None, :]
            # Reflections mostly see sides the capture never looked at (undersides), where
            # band 1 extrapolates to false colours: keep the splat's own colour, take only
            # the brightness from the light's direction.
            full = self.diffuse_at(-ldirs)
            lum_w = np.array([0.2126, 0.7152, 0.0722], np.float32)
            k = (full @ lum_w) / np.maximum(self.c @ lum_w, 1e-3)
            lin = lin + srgb_to_linear(np.maximum(self.c, 0.0) * np.clip(k, 0.0, 4.0)[:, None]) * w
        return ((np.maximum(linear_to_srgb(lin), 0.0) - 0.5) / C0).astype(np.float32)

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
        self.mesh_vis = {}
        self.last_vis = {}
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
            vis = self._shadow(eye, shadows, shadow_res, softness, report, L["name"], L)
            self.last_vis[L["name"]] = vis
            col = np.asarray(L["colour"], np.float32)[None, :]
            vis = np.broadcast_to(np.asarray(vis, np.float32), e.shape)
            w = (e * vis)[:, None] * col
            lin = lin + srgb_to_linear(self.diffuse_at(-ldirs)) * w
            if has_sh:
                h = v - ldirs
                h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
                # The light's strength goes through the display curve; the shadow scales the
                # highlight linearly (a curve on it would let 1% light through as 12%).
                ws = np.power(np.maximum(e[:, None] * col, 0), 1 / 2.2) * vis[:, None]
                shine += ws * self.shine_at(h)
        out = np.maximum(linear_to_srgb(lin) + shine, 0.0)
        return ((out - 0.5) / C0).astype(np.float32), None
