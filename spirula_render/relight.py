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


def _bilinear(u, v, res):
    """The four pixels around continuous pixel coordinates u, v and their weights:
    (iu [N,4], iv [N,4], w [N,4]). Pixel i covers [i, i+1); its centre is i + 0.5."""
    return _footprint(u, v, res, 1.0)


def _footprint(u, v, res, radius):
    """Pixels within `radius` of (u, v) and tent weights (1 - distance / radius per axis),
    normalised: (iu, iv, w), each [N, k*k]. radius 1 is bilinear (2x2); 1.5 spreads over
    3x3, a wider filter. Continuous in u, v: a pixel enters or leaves at zero weight."""
    n_ax = 2 if radius <= 1.0 else 3
    x, y = u - 0.5, v - 0.5
    if n_ax == 2:
        bx, by = np.floor(x), np.floor(y)
    else:
        bx, by = np.floor(x) - 1 + (x - np.floor(x) >= 0.5), np.floor(y) - 1 + (y - np.floor(y) >= 0.5)
    offs = np.arange(n_ax)
    cx = bx[:, None] + offs[None, :]                              # [N, n] tap pixel indices
    cy = by[:, None] + offs[None, :]
    wx = np.maximum(1.0 - np.abs(x[:, None] - cx) / radius, 0.0)
    wy = np.maximum(1.0 - np.abs(y[:, None] - cy) / radius, 0.0)
    wx /= np.maximum(wx.sum(1, keepdims=True), 1e-12)
    wy /= np.maximum(wy.sum(1, keepdims=True), 1e-12)
    iu = np.repeat(cx, n_ax, 1).astype(np.int64)                  # [N, n*n]: x fastest inner? (y-major)
    iv = np.tile(cy, (1, n_ax)).astype(np.int64)
    w = (np.repeat(wx, n_ax, 1) * np.tile(wy, (1, n_ax))).astype(np.float32)
    return np.clip(iu, 0, res - 1), np.clip(iv, 0, res - 1), w


class DeepShadow:
    """Light transmittance at every depth, seen from one light (a deep opacity map).

    Fur is semi-transparent: light fades gradually as it passes through hair, so a
    splat's visibility is the product of (1 - opacity) of everything between it and
    the light, not a single "is it behind the surface" test. Each splat's optical depth
    is shared among the four map pixels around its centre (bilinear weights), each
    pixel's entries are sorted by distance, and a running sum gives the optical depth in
    front of any depth there. Reading back uses the same four pixels and weights.

    Sharing (rather than dropping each splat into the one pixel under its centre) keeps
    shadows continuous as the light moves: a hair crossing a pixel boundary hands its
    shadow over gradually instead of all at once, which showed as jittering fur.
    """

    footprint = 1.0            # pixels each splat is shared over (1 bilinear, 1.5 = 3x3)

    def __init__(self, V, f, res, means, opacity, size, footprint=None):
        """V: world->light camera (OpenCV axes), f: focal length in pixels. Every splat
        given here must be in front of the light and inside its view."""
        self.V, self.f, self.res = V, f, res
        u, v, dist, z = self._project(means)
        # Share of a pixel each splat covers (its 2-sigma disc), so small splats
        # occlude a little and big ones fully.
        r_pix = 2.0 * size * self.f / np.maximum(z, 1e-9)
        cover = np.minimum(1.0, math.pi * r_pix * r_pix)
        a = np.clip(opacity * cover, 0.0, 0.999)
        tau = -np.log1p(-a).astype(np.float64)                         # optical depth
        self.dmax = (float(dist.max()) if len(dist) else 1.0) * 1.0001 + 1e-9
        if footprint is not None:
            self.footprint = footprint
        iu, iv, w = _footprint(u, v, res, self.footprint)
        self.iu, self.iv, self.w = iu, iv, w
        self.k = w.shape[1]
        pix = iv * res + iu                                            # [N, k]
        key = (pix.astype(np.float64) + (dist / self.dmax)[:, None]).ravel()
        order = np.argsort(key)
        self.key = key[order]
        tw = (tau[:, None] * w).ravel()[order]
        self.cum = np.concatenate([[0.0], np.cumsum(tw)])
        # for the soft depth test: running sum of optical depth x its position
        self.cumk = np.concatenate([[0.0], np.cumsum(tw * (self.key - np.floor(self.key)))])
        self.rank = np.empty(len(order), np.int64)                     # entry -> sorted place
        self.rank[order] = np.arange(len(order))
        self.rank = self.rank.reshape(-1, self.k)
        # where each pixel's entries start in the sorted order
        self.start = np.searchsorted(self.key, np.arange(res * res + 1, dtype=np.float64), side="left")
        self.pix = pix
        self.dist = dist
        self.u, self.v = u, v
        # Exact coverage per pixel from the renderer itself (every splat's full footprint),
        # seen from the light. The binned map above only knows splat centres, which
        # undercounts coverage through a thin shell; it is used for depth ORDER only.
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

    def _front(self, s, lo, hi, x_lo, x_hi):
        """Optical depth in front of a soft depth window [x_lo, x_hi] (key units, within the
        column starting at sorted index s; lo, hi = sorted indices of the window's ends).

        An occluder fully in front counts fully, one inside the window in proportion to
        how far into it the receiver is -- the box-filtered depth test. A hard test made a
        hair's whole shadow switch on or off as its depth crossed the receiver's, which
        showed as jittering fur when the light moved."""
        def G(stop, x):                     # integral of the in-front optical depth up to x
            C = self.cum[stop] - self.cum[s]
            K = self.cumk[stop] - self.cumk[s]
            return x * C - K
        width = np.maximum(x_hi - x_lo, 1e-12)
        return (G(hi, x_hi) - G(lo, x_lo)) / width

    def _coverage_log_T(self, u, v, softness):
        """The renderer's coverage seen from the light (log transmittance), box-blurred for
        soft edges and read bilinearly at u, v."""
        logT = _blurred_log_T(self.gpu_T, softness)
        iu, iv, w = _footprint(u, v, self.res, self.footprint)
        return np.einsum('nk,nk->n', logT[iv, iu], w)

    def _combine(self, front, whole, w, u, v, softness, empty_frac=None):
        """Per point: optical depth in front of it and in its whole column (per neighbour
        pixel, [N,4]) -> light reaching it."""
        f_tau = np.einsum('nk,nk->n', front, w)
        w_tau = np.einsum('nk,nk->n', whole, w)
        if self.gpu_T is None:
            return np.exp(-self.density * f_tau).astype(np.float32)
        frac = np.where(w_tau > 1e-12, f_tau / np.maximum(w_tau, 1e-12),
                        0.0 if empty_frac is None else empty_frac)
        lt = self._coverage_log_T(u, v, softness)
        return np.exp(self.density * np.clip(frac, 0.0, 1.0) * lt).astype(np.float32)

    def self_visibility(self, softness=1, bias=0.0):
        """Light reaching each splat this map was built from: the share of its columns'
        optical depth lying in front of it (more than `bias` in front, so a splat's own
        surface layer doesn't shade it), applied to the renderer's coverage there."""
        r = self.rank                                                  # [N, 4]
        s = self.start[self.pix]
        e = self.start[self.pix + 1]
        frac_key = self.key - np.floor(self.key)
        if np.any(np.asarray(bias) > 0):
            # soft depth window [own depth - 2 bias, own depth): searched in sorted order,
            # the queries are nearly ascending, which numpy's binary search exploits
            n = len(self.key)
            inv = np.empty(n, np.int64)
            inv[r.ravel()] = np.arange(n) // self.k                     # sorted place -> splat
            b = np.asarray(bias, np.float64)
            b_sorted = b[inv] if b.ndim else np.full(n, float(b))
            q = np.maximum(self.key - 2 * b_sorted / self.dmax, np.floor(self.key))
            lo_sorted = np.minimum(np.searchsorted(self.key, q, side="left"), np.arange(n))
            lo = lo_sorted[r]
            x_lo = (q - np.floor(q))[r]
            x_hi = frac_key[r]
            front = self._front(s, lo, r, x_lo, x_hi)
        else:
            front = self.cum[r] - self.cum[s]
        whole = self.cum[e] - self.cum[s]
        return self._combine(front, whole, self.w, self.u, self.v, softness)

    def query(self, pts, bias, softness, surface=False):
        """Light reaching arbitrary points (other splats, mesh surfaces). Points outside the
        light's view are lit. surface=True: a column without splats takes its in-front
        share from comparing the point's distance with the coverage's mean depth around it
        (its coverage there is only neighbours' blurred edges)."""
        u, v, dist, _ = self._project(pts)
        pz = pts @ self.V[2, :3].astype(np.float32) + np.float32(self.V[2, 3])
        res = self.res
        seen = (pz > 1e-9) & (u >= 0) & (u < res) & (v >= 0) & (v < res)
        out = np.ones(len(pts), np.float32)
        if not np.any(seen):
            return out
        u, v, dist = u[seen], v[seen], dist[seen]
        iu, iv, w = _footprint(u, v, res, self.footprint)
        pix = iv * res + iu
        b = np.asarray(bias)[seen] if np.ndim(bias) else bias
        # soft depth window [dist - 2 bias, dist) (the hard test was at dist - bias)
        d_hi = np.minimum(np.maximum(dist, 0.0) / self.dmax, 0.9999)
        d_lo = np.minimum(np.maximum(dist - 2 * b, 0.0) / self.dmax, d_hi)
        s = self.start[pix]
        e = self.start[pix + 1]
        lo = np.searchsorted(self.key, (pix + d_lo[:, None]).ravel(), side="left").reshape(pix.shape)
        hi = np.searchsorted(self.key, (pix + d_hi[:, None]).ravel(), side="left").reshape(pix.shape)
        x_lo = np.broadcast_to(d_lo[:, None], pix.shape)
        x_hi = np.broadcast_to(d_hi[:, None], pix.shape)
        front = self._front(s, lo, hi, x_lo, x_hi)
        whole = self.cum[e] - self.cum[s]
        empty_frac = None
        if surface and self.gpu_depth is not None:
            r = softness + 1
            k = 2 * r + 1
            a = self.gpu_alpha.astype(np.float64)

            def box(img):
                for axis in (0, 1):
                    c = np.cumsum(np.pad(img, [(r + 1, r)] if axis == 0 else [(0, 0), (r + 1, r)],
                                         mode='constant'), axis=axis)
                    img = (c[k:] - c[:-k]) if axis == 0 else (c[:, k:] - c[:, :-k])
                return img
            D = box(a * self.gpu_depth) / np.maximum(box(a), 1e-9)
            Dp = np.einsum('nk,nk->n', D[iv, iu], w)
            b = float(np.max(bias)) if np.ndim(bias) else float(bias)
            empty_frac = np.clip((dist - Dp) / max(4.0 * b, 1e-9) + 0.5, 0.0, 1.0)
        out[seen] = self._combine(front, whole, w, u, v, softness, empty_frac)
        return out

    def visibility(self, means, bias, radius=1):
        return self.query(means, bias, radius)


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
    """Light reaching points on other surfaces (meshes) through the splats of map `sm`."""
    return sm.query(pts, bias, softness, surface=True)


def light_visibility(eye, target, corners, means, opacity, size, query, bias, res, softness,
                     density=1.0, tris=None, extra=None, extra_bias=0.0, footprint=1.0):
    """See _light_visibility; cross-faded between shadow-map layouts (_view_modes)."""
    return _blend_modes(lambda mode: _light_visibility(
        eye, target, corners, means, opacity, size, query, bias, res, softness, density, tris,
        extra, extra_bias, mode, footprint), eye, target, corners)


def _light_visibility(eye, target, corners, means, opacity, size, query, bias, res, softness,
                      density=1.0, tris=None, extra=None, extra_bias=0.0, mode="single",
                      footprint=1.0):
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

    for V, f_unit, fi in _face_views(eye, target, corners, mode):
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
        sm = DeepShadow(V, f, res, means[inside], opacity[inside], size[inside], footprint)
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


SINGLE_UNTIL, CUBE_FROM = 1.4, 1.8      # map half-width (tan) where single fades into cube


def _view_modes(eye, target, corners):
    """How to shadow from a light at `eye`: [(mode, weight)]. One map aimed at the model
    while it fits a moderate field of view, a cube of six maps around a light close to or
    inside it, and in between both, cross-faded: the two disagree slightly (a cube face
    spreads its pixels wider), and a hard switch showed as a jump in brightness."""
    V = look_at(eye, target)
    pc = corners @ V[:3, :3].T + V[:3, 3]
    if not np.all(pc[:, 2] > 1e-9):
        return [("cube", 1.0)]
    t = float(np.max(np.abs(pc[:, :2]) / pc[:, 2:3]))
    if t <= SINGLE_UNTIL:
        return [("single", 1.0)]
    if t >= CUBE_FROM:
        return [("cube", 1.0)]
    x = (t - SINGLE_UNTIL) / (CUBE_FROM - SINGLE_UNTIL)
    w = x * x * (3 - 2 * x)
    return [("single", 1.0 - w), ("cube", w)]


def _blend_modes(fn, eye, target, corners):
    """Run fn(mode) for each mode the light needs and cross-fade the results (arrays, or
    tuples of arrays)."""
    out = None
    for mode, w in _view_modes(eye, target, corners):
        r = fn(mode)
        if out is None:
            out = tuple(x * np.float32(w) for x in r) if isinstance(r, tuple) else r * np.float32(w)
        else:
            out = tuple(a + b * np.float32(w) for a, b in zip(out, r)) if isinstance(r, tuple) \
                else out + r * np.float32(w)
    return out


def _face_views(eye, target, corners, mode="single"):
    """The light's view(s) for `mode`: one map aimed at the model ("single"), or a cube of
    six around the light ("cube"). Yields (V, f_per_unit_res, face_index or None)."""
    if mode == "single":
        V = look_at(eye, target)
        pc = corners @ V[:3, :3].T + V[:3, 3]
        t = float(np.max(np.abs(pc[:, :2]) / np.maximum(pc[:, 2:3], 1e-9)))
        yield V, 0.5 / (t * 1.05), None
        return
    ups = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 1, 0), (0, 1, 0)]
    for axis in range(3):
        for sign_i, sign in enumerate((1.0, -1.0)):
            d = np.zeros(3); d[axis] = sign
            yield look_at(eye, np.asarray(eye, np.float64) + d, ups[axis * 2 + sign_i]), 0.5 / 1.05, axis * 2 + sign_i


def quick_light_visibility(eye, target, corners, means, width, res, density=1.0, tris=None,
                           bias=0.0, extra=None):
    """See _quick_light_visibility; cross-faded between shadow-map layouts (_view_modes)."""
    return _blend_modes(lambda mode: _quick_light_visibility(
        eye, target, corners, means, width, res, density, tris, bias, extra, mode),
        eye, target, corners)


def _quick_light_visibility(eye, target, corners, means, width, res, density=1.0, tris=None,
                            bias=0.0, extra=None, mode="single"):
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
        iu, iv, wb = _bilinear(u, v, res)        # four pixels, blended: no jumps as it moves
        dist = np.linalg.norm(pc, axis=1)[:, None]
        a = alpha[iv, iu]
        med = np.where(a > 0.5, median[iv, iu], np.inf)
        ww = np.asarray(w)[:, None] if np.ndim(w) else w
        # light is 50% at the median depth by definition, fades from full in front of it,
        # and can't fall below what gets through the whole column
        T_d = np.exp(-math.log(2.0) * np.exp(np.clip((dist - med) / np.maximum(ww, 1e-12), -30, 30)))
        T_floor = np.power(np.clip(1.0 - a, 1e-6, 1.0), density)
        out = np.einsum('nk,nk->n', np.maximum(T_d, T_floor), wb)
        if outside_lit:
            out = np.where((pc[:, 2] > 1e-9) & (u >= 0) & (u < res) & (v >= 0) & (v < res), out, 1.0)
        return out

    for V, f_unit, fi in _face_views(eye, target, corners, mode):
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

    def shadow_bias(self):
        """How far in front of a splat an occluder must be to shade it (the soft depth
        test's window is twice this): the splat's own size, but at least a hair's
        thickness, or tiny splats flip between lit and shadowed as near-equal depths swap."""
        return np.maximum(2.0 * self.splat_size, np.float32(0.002 * self.extent))

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
                               self.occluder_size, self.means, self.shadow_bias(),
                               shadow_res, softness, self.shadow_density,
                               self._tris_for(L) if L is not None else self.mesh_tris,
                               self.mesh_points, 0.004 * self.extent, self.shadow_footprint)
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
                                           self.shadow_bias(), self.mesh_points)
                if self.mesh_points is not None:
                    v, self.mesh_vis[L["name"]] = v
                e = e * v
            w = np.power(e, np.float32(1 / 2.2))[:, None] * \
                np.asarray(L["colour"], np.float32)[None, :]
            out = out + self.diffuse_at(-ldirs) * w
        return ((out - 0.5) / C0).astype(np.float32)

    shine_colour = 0.25        # 0: neutral (brightness only) ... 1: the full captured colour
    shadow_density = 4.0       # see SpirulaSettings.shadow_density
    shadow_footprint = 1.5     # shadow-map pixels each splat is spread over (DeepShadow)
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
