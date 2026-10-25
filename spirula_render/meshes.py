"""Polygon objects alongside the splats.

Meshes are path traced by Blender's own Cycles, run inside this engine: it renders them
(materials, textures, mesh-on-mesh shadows and reflections) into this engine's render
result, with a few extra passes -- the 3D position under each pixel and each light's share
of the light (a light group per light). The splats then join in:

* splats shadow meshes: each light's share is dimmed by the splat shadow map at that
  pixel's 3D position (the same deep opacity maps the splats shadow themselves with);
* meshes shadow splats: the meshes' triangles are drawn into a depth map from each light;
* meshes and splats meet per pixel by depth.

Cycles never sees the splats (they're hidden from all its rays while it renders), and its
lights are converted to this engine's units: a light whose irradiance equals the capture
light's gives a white diffuse surface the brightness 1, as it gives a splat its scanned
colour; in Cycles that surface comes out 1/pi.
"""

import math

import bpy
import numpy as np

from . import scene_data as sd

GEOMETRY = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}
LIGHTGROUP = "spirula_%d"
ON_TOP_INDEX = 7351           # object index marking "always on top" meshes during a render
CYCLES_UNIT = math.pi          # this engine's light units / Cycles' (see the module notes)


def mesh_instances(depsgraph):
    """(evaluated object, world matrix) for every visible polygon object, instances included."""
    out = []
    for inst in depsgraph.object_instances:
        ob = inst.object
        if ob.type in GEOMETRY and not sd.is_splat_object(ob):
            out.append((ob, np.array(inst.matrix_world, np.float64)))
    return out


def instances_key(depsgraph):
    """Which meshes are where: changes when one is added, removed or moved."""
    return tuple((ob.name, tuple(np.round(M, 6).ravel())) for ob, M in mesh_instances(depsgraph))


def has_meshes(depsgraph):
    return any(inst.object.type in GEOMETRY and not sd.is_splat_object(inst.object)
               for inst in depsgraph.object_instances)


def world_triangles(depsgraph):
    """All shadow-casting mesh triangles in world space: [T, 3, 3] float32."""
    parts = []
    for ob, M in mesh_instances(depsgraph):
        if not getattr(ob, "visible_shadow", True):
            continue
        try:
            me = ob.to_mesh()
        except RuntimeError:
            continue
        try:
            me.calc_loop_triangles()
            nv, nt = len(me.vertices), len(me.loop_triangles)
            if nt == 0:
                continue
            co = np.empty(nv * 3, np.float32)
            me.vertices.foreach_get("co", co)
            tri = np.empty(nt * 3, np.int32)
            me.loop_triangles.foreach_get("vertices", tri)
        finally:
            ob.to_mesh_clear()
        co = co.reshape(-1, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
        parts.append(co[tri.reshape(-1, 3)].astype(np.float32))
    if not parts:
        return None
    return np.concatenate(parts)


def _clip_near(tc, near):
    """Clip camera-space triangles [T,3,3] against z >= near. Returns triangles [T',3,3]."""
    z = tc[:, :, 2]
    inside = z >= near
    n_in = inside.sum(1)
    keep = [tc[n_in == 3]]
    # one vertex inside: one smaller triangle; two inside: a quad, two triangles
    for count in (1, 2):
        sel = tc[n_in == count]
        if not len(sel):
            continue
        ins = inside[n_in == count]
        # rotate each triangle so the odd vertex (the lone in/out one) comes first
        odd = np.argmax(ins if count == 1 else ~ins, axis=1)
        idx = (odd[:, None] + np.arange(3)[None, :]) % 3
        t = np.take_along_axis(sel, idx[:, :, None], 1)
        a, b, c = t[:, 0], t[:, 1], t[:, 2]

        def cut(p, q):
            s = ((near - p[:, 2]) / (q[:, 2] - p[:, 2]))[:, None]
            return p + (q - p) * s
        ab, ac = cut(a, b), cut(a, c)
        if count == 1:
            keep.append(np.stack([a, ab, ac], 1))
        else:
            keep.append(np.stack([ab, b, c], 1))
            keep.append(np.stack([ab, c, ac], 1))
    return np.concatenate(keep) if keep else tc[:0]


def rasterize_distance(V, f, res, tris):
    """Distance from the eye to the nearest triangle at each pixel of a res x res pinhole view
    (V world->camera with OpenCV axes, principal point at the centre). inf where empty."""
    out = np.full(res * res, np.inf, np.float32)
    if tris is None or not len(tris):
        return out.reshape(res, res)
    R, t = np.asarray(V, np.float64)[:3, :3], np.asarray(V, np.float64)[:3, 3]
    tc = tris.astype(np.float64) @ R.T + t
    tc = _clip_near(tc, 1e-6)
    if not len(tc):
        return out.reshape(res, res)
    c = res / 2
    u = f * tc[:, :, 0] / tc[:, :, 2] + c - 0.5           # pixel-centre coordinates
    v = f * tc[:, :, 1] / tc[:, :, 2] + c - 0.5
    iz = 1.0 / tc[:, :, 2]
    lo_u = np.clip(np.floor(u.min(1)), 0, res - 1).astype(np.int64)
    hi_u = np.clip(np.ceil(u.max(1)), 0, res - 1).astype(np.int64)
    lo_v = np.clip(np.floor(v.min(1)), 0, res - 1).astype(np.int64)
    hi_v = np.clip(np.ceil(v.max(1)), 0, res - 1).astype(np.int64)
    on = (u.max(1) >= 0) & (u.min(1) <= res - 1) & (v.max(1) >= 0) & (v.min(1) <= res - 1)
    w, h = hi_u - lo_u + 1, hi_v - lo_v + 1

    def raster(sel, gu, gv):
        """Triangles `sel` against candidate pixels gu, gv [n, k] (pixel coords)."""
        uu, vv, zz = u[sel], v[sel], iz[sel]
        x0, y0 = uu[:, 0:1], vv[:, 0:1]
        e1u, e1v = uu[:, 1:2] - x0, vv[:, 1:2] - y0
        e2u, e2v = uu[:, 2:3] - x0, vv[:, 2:3] - y0
        den = e1u * e2v - e2u * e1v
        ok_t = np.abs(den) > 1e-12
        den = np.where(ok_t, den, 1.0)
        pu, pv = gu - x0, gv - y0
        b1 = (pu * e2v - e2u * pv) / den
        b2 = (e1u * pv - pu * e1v) / den
        b0 = 1 - b1 - b2
        eps = -1e-6
        hit = ok_t & (b0 >= eps) & (b1 >= eps) & (b2 >= eps) & \
            (gu >= 0) & (gu <= res - 1) & (gv >= 0) & (gv <= res - 1)
        inv_z = b0 * zz[:, 0:1] + b1 * zz[:, 1:2] + b2 * zz[:, 2:3]   # 1/z is linear on screen
        z = 1.0 / np.maximum(inv_z, 1e-12)
        x = (gu + 0.5 - c) / f
        y = (gv + 0.5 - c) / f
        dist = z * np.sqrt(1 + x * x + y * y)
        pix = (gv.astype(np.int64) * res + gu.astype(np.int64))
        np.minimum.at(out, pix[hit], dist[hit].astype(np.float32))

    # small triangles together, a fixed block of candidate pixels each
    B = 8
    small = np.nonzero(on & (w <= B) & (h <= B))[0]
    if len(small):
        oy, ox = np.mgrid[0:B, 0:B]
        for s in range(0, len(small), 20000):
            sel = small[s:s + 20000]
            gu = (lo_u[sel][:, None] + ox.ravel()[None, :]).astype(np.float64)
            gv = (lo_v[sel][:, None] + oy.ravel()[None, :]).astype(np.float64)
            raster(sel, gu, gv)
    # large ones one at a time over their bounding box
    for i in np.nonzero(on & ~((w <= B) & (h <= B)))[0]:
        gv, gu = np.mgrid[lo_v[i]:hi_v[i] + 1, lo_u[i]:hi_u[i] + 1]
        raster(np.array([i]), gu.reshape(1, -1).astype(np.float64),
               gv.reshape(1, -1).astype(np.float64))
    return out.reshape(res, res)


def mesh_visibility(dist_map, V, f, res, pts, bias, softness):
    """Light reaching points `pts` past the meshes (0..1), from a light's distance map,
    averaged over (2*softness+1)^2 pixels for soft edges. Points outside the map: lit."""
    R, t = np.asarray(V, np.float32)[:3, :3], np.asarray(V, np.float32)[:3, 3]
    pc = pts @ R.T + t
    z = pc[:, 2]
    front = z > 1e-9
    zs = np.maximum(z, 1e-9)
    u = f * pc[:, 0] / zs + res / 2
    v = f * pc[:, 1] / zs + res / 2
    from .relight import _bilinear
    dist = np.linalg.norm(pc, axis=1)
    inside = front & (u >= 0) & (u < res) & (v >= 0) & (v < res)
    total = np.zeros(len(pts), np.float32)
    taps = 0
    b = (np.asarray(bias)[:, None] if np.ndim(bias) else bias)
    for du in range(-softness, softness + 1):
        for dv in range(-softness, softness + 1):
            # each tap blended over its four pixels, so the edge moves smoothly with the light
            iu, iv, w = _bilinear(u + du, v + dv, res)
            lit = dist_map[iv, iu] >= dist[:, None] - b
            total += np.einsum('nk,nk->n', lit.astype(np.float32), w)
            taps += 1
    return np.where(inside, total / taps, 1.0).astype(np.float32)


# ---------------------------------------------------------------- Cycles setup per render
_SAVED = {}


def _set(owner, attr, value, store):
    store.append((owner, attr, getattr(owner, attr)))
    setattr(owner, attr, value)


def prepare_cycles(scene, view_layer):
    """Before a render with meshes: hide the splats from Cycles, make the film transparent
    (the mesh layer needs its own alpha), switch on the passes we read, and give every
    light its own light group. Everything is put back by restore_cycles."""
    restore_cycles()
    store = []
    lights = []
    try:
        for ob in scene.objects:
            if ob.type == 'POINTCLOUD' and sd.is_splat_object(ob):
                for attr in ("visible_camera", "visible_diffuse", "visible_glossy",
                             "visible_transmission", "visible_volume_scatter", "visible_shadow"):
                    _set(ob, attr, False, store)
        _set(scene.render, "film_transparent", True, store)
        # We read Cycles' finished passes back in one piece, and they come back empty when
        # Cycles splits the render into tiles (anything over the 2048 default): make one
        # tile cover the whole image. Cycles allows tiles up to 8192.
        r = scene.render
        need = max(r.resolution_x, r.resolution_y) * r.resolution_percentage // 100
        _set(scene.cycles, "use_auto_tile", True, store)
        if scene.cycles.tile_size < need:
            _set(scene.cycles, "tile_size", min(need, 8192), store)
        if need > 8192:
            print("Spirula: renders over 8192 px can't fit one Cycles tile; meshes may be missing")
        # meshes are rendered pinhole: depth of field is done for meshes and splats together
        cam = scene.camera
        if cam is not None and cam.type == 'CAMERA':
            _SAVED["dof"] = (cam.data.name, cam.data.dof.use_dof)
            _set(cam.data.dof, "use_dof", False, store)
        _set(view_layer, "use_pass_z", True, store)
        _set(view_layer, "use_pass_position", True, store)
        _set(view_layer, "use_pass_diffuse_color", True, store)
        _set(view_layer, "use_pass_glossy_color", True, store)
        _set(view_layer, "use_pass_glossy_direct", True, store)
        _set(view_layer, "use_pass_glossy_indirect", True, store)
        _set(view_layer, "use_pass_object_index", True, store)
        for ob in scene.objects:
            if ob.type in GEOMETRY:
                _set(ob, "pass_index", ON_TOP_INDEX if getattr(ob, "spirula_on_top", False) else 0,
                     store)
        groups = view_layer.lightgroups
        for ob in scene.objects:
            if ob.type == 'LIGHT':
                name = LIGHTGROUP % len(lights)
                if groups.get(name) is None:
                    groups.add(name=name)
                    store.append((groups, "#remove", name))
                _set(ob, "lightgroup", name, store)
                lights.append(ob.name)
    except Exception:
        _SAVED["store"] = store
        restore_cycles()
        raise
    _SAVED["store"] = store
    _SAVED["lights"] = lights
    return lights


def restore_cycles():
    store = _SAVED.pop("store", None)
    _SAVED.pop("lights", None)
    _SAVED.pop("dof", None)
    if not store:
        return
    for owner, attr, old in reversed(store):
        try:
            if attr == "#remove":
                g = owner.get(old)
                if g is not None:
                    owner.remove(g)
            else:
                setattr(owner, attr, old)
        except (ReferenceError, AttributeError, TypeError):
            pass


def light_groups():
    """Light object name -> its light group pass name, for the render in progress."""
    return {name: "Combined_" + LIGHTGROUP % i for i, name in enumerate(_SAVED.get("lights", []))}


def read_passes(result):
    """Every pass of a render result as float32 arrays [H, W, C], top row first."""
    W, H = result.resolution_x, result.resolution_y
    out = {}
    for p in result.layers[0].passes:
        a = np.empty(W * H * p.channels, np.float32)
        p.rect.foreach_get(a)
        out[p.name] = a.reshape(H, W, p.channels)[::-1]
    return out


def original_use_dof(cam_data):
    saved = _SAVED.get("dof")
    if saved is not None and saved[0] == cam_data.name:
        return saved[1]
    return cam_data.dof.use_dof


def prepared():
    return bool(_SAVED.get("store")) or "lights" in _SAVED


def original_film_transparent(scene):
    return _SAVED.get("film_transparent", scene.render.film_transparent)


def _scene_has_meshes(scene, view_layer):
    return any(ob.type in GEOMETRY and not ob.hide_render and not sd.is_splat_object(ob)
               for ob in view_layer.objects)


@bpy.app.handlers.persistent
def _render_pre(scene, *_):
    if scene.render.engine != 'SPIRULA' or not scene.spirula.use_meshes:
        return
    view_layer = bpy.context.view_layer if bpy.context.view_layer else scene.view_layers[0]
    if not _scene_has_meshes(scene, view_layer):
        return
    film = scene.render.film_transparent
    prepare_cycles(scene, view_layer)
    _SAVED["film_transparent"] = film


@bpy.app.handlers.persistent
def _render_post(scene, *_):
    restore_cycles()
    _SAVED.pop("film_transparent", None)


def register():
    for h, fn in ((bpy.app.handlers.render_pre, _render_pre),
                  (bpy.app.handlers.render_post, _render_post),
                  (bpy.app.handlers.render_cancel, _render_post)):
        if fn not in h:
            h.append(fn)


def unregister():
    restore_cycles()
    for h, fn in ((bpy.app.handlers.render_pre, _render_pre),
                  (bpy.app.handlers.render_post, _render_post),
                  (bpy.app.handlers.render_cancel, _render_post)):
        if fn in h:
            h.remove(fn)


def _roughness(mat):
    nt = getattr(mat, "node_tree", None) if mat is not None else None
    if nt is not None:
        for node in nt.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                sock = node.inputs.get("Roughness")
                if sock is not None and not sock.is_linked:
                    return float(sock.default_value)
                return 0.5
    return 0.5


def _reflectance(mat):
    """Normal-incidence reflectance (rgb) of a material: metals reflect their base colour,
    other materials about 4%."""
    nt = getattr(mat, "node_tree", None) if mat is not None else None
    if nt is not None:
        for node in nt.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bc, mt = node.inputs.get("Base Color"), node.inputs.get("Metallic")
                base = np.array(bc.default_value[:3] if bc is not None and not bc.is_linked
                                else (0.8, 0.8, 0.8), np.float64)
                metal = float(mt.default_value) if mt is not None and not mt.is_linked else 0.0
                return metal * base + (1 - metal) * 0.04
    return np.full(3, 0.04)


def _hull(pts):
    """Convex hull of 2D points, counter-clockwise (monotone chain)."""
    pts = sorted(set(map(tuple, np.round(pts, 9))))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def mirror_planes(depsgraph, limit=2):
    """Flat polygon objects that may reflect the splats, largest first: dicts with the plane
    (n, off), roughness, name, reflectance (rgb), and its outline as a convex polygon in the
    plane (origin, u, v axes and 2D corners). How strongly each pixel reflects comes from
    Cycles' Glossy Color pass; the outline and reflectance are for the light it bounces."""
    found = []
    for ob, M in mesh_instances(depsgraph):
        try:
            me = ob.to_mesh()
        except RuntimeError:
            continue
        try:
            nv = len(me.vertices)
            if nv < 3 or nv > 20000:
                continue
            co = np.empty(nv * 3, np.float32)
            me.vertices.foreach_get("co", co)
            mat = ob.material_slots[0].material if ob.material_slots else None
            rough = _roughness(mat)
            refl = _reflectance(mat)
        finally:
            ob.to_mesh_clear()
        P = co.reshape(-1, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
        c = P.mean(0)
        _, sv, Vt = np.linalg.svd(P - c, full_matrices=False)
        if sv[0] <= 0 or sv[-1] / sv[0] > 1e-5:
            continue
        n = Vt[-1] / np.linalg.norm(Vt[-1])
        u, v = Vt[0], Vt[1]
        hull = _hull(np.stack([(P - c) @ u, (P - c) @ v], 1))
        found.append((sv[0] * sv[1], dict(n=n, off=float(n @ c), rough=rough, name=ob.name,
                                          reflectance=refl, origin=c, u=u, v=v, outline=hull)))
    found.sort(key=lambda t: -t[0])
    return [t[1] for t in found[:limit]]


def mirror_lights(lights, mirrors, min_reflectance=0.2):
    """Each light's image in each mirror (flat surface reflecting at least min_reflectance):
    light that bounces off the mirror onto the splats. The image sits behind the mirror,
    dimmed by its reflectance, and only shines through the mirror's outline (its "portal")."""
    lum = np.array([0.2126, 0.7152, 0.0722])
    out = []
    for m in mirrors:
        r = np.asarray(m["reflectance"], np.float64)
        if float(r @ lum) < min_reflectance:
            continue
        n, off = np.asarray(m["n"], np.float64), m["off"]
        for L in lights:
            if "portal" in L:
                continue
            V = dict(L)
            d = np.asarray(L["dir"], np.float64)
            if L["type"] == 'SUN':
                side = -float(n @ d)
            else:
                p = np.asarray(L["pos"], np.float64)
                side = float(n @ p - off)
                V["pos"] = p - 2 * side * n
            if abs(side) < 1e-9:
                continue
            V["dir"] = d - 2 * float(n @ d) * n
            V["colour"] = tuple(np.asarray(L["colour"], np.float64) * r)
            V["name"] = f'{L["name"]} in {m["name"]}'
            V["portal"] = dict(n=n, off=off, side=float(np.sign(side)), origin=m["origin"],
                               u=m["u"], v=m["v"], outline=m["outline"])
            out.append(V)
    return out


def portal_mask(portal, pts, light):
    """Which points the light through a mirror reaches: on the mirror's lit side, with the
    path from the light's image to them crossing the plane inside the mirror's outline."""
    n = np.asarray(portal["n"], np.float32)
    off = np.float32(portal["off"])
    ok = (pts @ n - off) * np.float32(portal["side"]) > 0
    if light["type"] == 'SUN':
        back = -np.asarray(light["dir"], np.float32)          # towards the light's image
        denom = float(back @ n)
        if abs(denom) < 1e-9:
            return np.zeros(len(pts), bool)
        t = (off - pts @ n) / np.float32(denom)
        X = pts + t[:, None] * back[None, :]
    else:
        q = np.asarray(light["pos"], np.float32)
        d = pts - q[None, :]
        denom = d @ n
        t = (off - q @ n) / np.where(np.abs(denom) < 1e-12, np.float32(1e-12), denom)
        ok &= (t > 0) & (t < 1)
        X = q[None, :] + t[:, None] * d
    rel = X - np.asarray(portal["origin"], np.float32)
    a = rel @ np.asarray(portal["u"], np.float32)
    b = rel @ np.asarray(portal["v"], np.float32)
    hull = portal["outline"]
    for i in range(len(hull)):
        x0, y0 = hull[i]
        x1, y1 = hull[(i + 1) % len(hull)]
        ok &= (x1 - x0) * (b - y0) - (y1 - y0) * (a - x0) >= -1e-9
    return ok
