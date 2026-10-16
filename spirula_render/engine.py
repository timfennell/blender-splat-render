"""The Spirula render engine: final renders (F12, animation) and the viewport's Rendered mode."""

import itertools
import math
import threading
import time

import bpy
import gpu
import numpy as np
from gpu_extras.presets import draw_texture_2d

from . import meshes, native, relight, scene_data as sd, transform, viewport_meshes

# One renderer (one GPU context and one uploaded splat set) serves the viewport and
# final renders, which can run at the same time on different threads.
_LOCK = threading.RLock()
_CACHE = {"key": None, "colour_key": None}
# Mesh layer for the render in progress: triangles that cast shadows on the splats, and
# the mesh surface points (one per pixel) the splats' shadows are wanted at.
_MESH = {"tris": None, "points": None, "serial": 0}
_SERIAL = itertools.count(1)


def _splat_key(obj):
    pc = obj.data
    return (pc.session_uid, len(pc.points), int(pc.get("gsp_revision", 0)),
            obj.name, tuple(a.name for a in pc.attributes))


_IDENTITY = np.eye(4)
_OBJECTS = {}           # object name -> (data key, matrix key, world-space arrays)


def _upload_scene(objs, report):
    """Merge every splat object into one world-space set and upload it if anything changed.

    Each object is moved into world space (positions, rotations, scales and its SH rotated
    with it) once per change of its data or transform, so all models sort, overlap and
    shadow each other in a single render."""
    keys = []
    parts = []
    for ob in objs:
        original = bpy.data.objects.get(ob.name) or ob     # attribute data lives here
        dkey = _splat_key(original)
        mkey = tuple(np.round(np.array(ob.matrix_world, np.float64), 7).ravel())
        cached = _OBJECTS.get(ob.name)
        if cached is None or cached[0] != dkey or cached[1] != mkey:
            local = sd.splat_arrays(original.data)
            world = transform.to_world(local, ob.matrix_world)
            if world.get("nonuniform"):
                report(f"Spirula: '{ob.name}' has a non-uniform scale; using its average scale")
            cached = (dkey, mkey, world)
            _OBJECTS[ob.name] = cached
        keys.append((ob.name, dkey, mkey))
        parts.append(cached[2])
    for name in list(_OBJECTS):
        if name not in {ob.name for ob in objs}:
            del _OBJECTS[name]
    key = tuple(keys)
    if _CACHE["key"] == key:
        return False
    d = parts[0] if len(parts) == 1 else transform.merge(parts)
    native.set_splats(d["means"], d["quats"], d["log_scales"], d["logit_opacities"],
                      d["dc"], d["sh"], d["degree"])
    _CACHE.update(key=key, n=d["n"], splats=d, relighter=None, colour_key=None)
    return True


def _object_space_lights(depsgraph, obj_world):
    """Blender lights, moved into the splat object's space."""
    Minv = np.linalg.inv(np.array(obj_world, np.float64))
    per_obj = float(np.mean(np.linalg.norm(np.array(obj_world)[:3, :3], axis=0)))
    out = []
    for inst in depsgraph.object_instances:
        ob = inst.object
        if ob.type != 'LIGHT':
            continue
        L = ob.data
        Mw = np.array(inst.matrix_world, np.float64)
        pos = (Minv @ Mw[:, 3])[:3]
        d = Minv[:3, :3] @ (Mw[:3, :3] @ np.array([0.0, 0.0, -1.0]))
        d /= np.linalg.norm(d)
        light = dict(name=ob.name, type=L.type if L.type in ('SUN', 'SPOT') else 'POINT',
                     pos=pos, dir=d, colour=tuple(L.color), world_per_object=per_obj,
                     strength=float(L.energy), power=float(L.energy))
        if L.type == 'SPOT':
            half = L.spot_size / 2
            light["spot_cos"] = (math.cos(half), math.cos(half * (1 - L.spot_blend)))
        out.append(light)
    return out


def _lights_key(lights, settings, shadows):
    rows = []
    for L in lights:
        rows.append((L["type"], tuple(np.round(L["pos"], 6)), tuple(np.round(L["dir"], 6)),
                     tuple(np.round(L["colour"], 4)), round(L["power"], 4), L.get("spot_cos")))
    return (tuple(rows), round(settings.ambient, 5), round(settings.exposure, 4),
            round(settings.shine_colour, 4), round(settings.shadow_density, 4),
            shadows, settings.shadow_resolution, settings.shadow_softness)


def _ensure_colours(depsgraph, obj, cam_world, settings, report, follow_camera=True, quick=False,
                    viewport=False):
    """Put the right colours on the GPU: the original ones, or relit by the scene's lights.

    follow_camera=False (viewport) keeps the last relit colours while only the view
    moves; lighting is redone when lights or settings change. quick=True relights the
    base colour only, without shadows, and renders without the higher SH bands: fast
    enough to follow a light being dragged. Returns the lighting key now on the GPU."""
    d = _CACHE["splats"]
    if not settings.use_relight:
        if _CACHE["colour_key"] is not None:
            native.set_colors(d["dc"], d["sh"])
            _CACHE["gpu_colours"] = (d["dc"], d["sh"])
            _CACHE["colour_key"] = None
        native.set_render_degree(-1)
        return None
    if _CACHE.get("relighter") is None:
        _CACHE["relighter"] = relight.Relighter(d)
    _CACHE["relighter"].shine_colour = settings.shine_colour
    _CACHE["relighter"].shadow_density = settings.shadow_density
    _CACHE["relighter"].mesh_tris = _MESH["tris"]
    _CACHE["relighter"].mesh_points = _MESH["points"]
    lights = _object_space_lights(depsgraph, _IDENTITY)
    Minv = _IDENTITY
    cam_pos = (Minv @ np.array(cam_world, np.float64)[:, 3])[:3].astype(np.float32)
    shadows = settings.use_shadows and (not viewport or settings.viewport_shadows) and not quick
    key = _lights_key(lights, settings, shadows or (quick and settings.viewport_shadows)) + \
        ("quick" if quick else "full",)
    if follow_camera:
        key = key + (tuple(np.round(cam_pos, 5)),)
    if _MESH["tris"] is not None or _MESH["points"] is not None:
        key = key + (_MESH["serial"],)
    if quick and _MESH["tris"] is not None:
        key = key + ("mesh shadows",)
    if _CACHE["colour_key"] != key and quick:
        lod_shadows = settings.use_shadows and settings.viewport_shadows
        qdc = _CACHE["relighter"].quick_dc(
            lights, cam_pos, settings.ambient, settings.exposure, lod_shadows,
            max(256, settings.shadow_resolution // 2))
        native.set_colors(qdc, None)
        _CACHE["gpu_colours"] = (qdc, None)
        _CACHE["colour_key"] = key
    elif _CACHE["colour_key"] != key:
        dc, sh = _CACHE["relighter"].colours(
            lights, cam_pos, settings.ambient, settings.exposure, shadows,
            settings.shadow_resolution, settings.shadow_softness, settings.primitive, report,
            diffuse_only=quick)
        native.set_colors(dc, sh)
        _CACHE["gpu_colours"] = (dc, sh)
        _CACHE["colour_key"] = key
    # Relit colours are baked per splat for this camera: render without the higher bands.
    native.set_render_degree(0)
    return _lights_key(lights, settings, settings.use_shadows)


def _freeze(depsgraph):
    """A copy of what the splat render reads from the depsgraph: splat objects, lights and their
    matrices. Cycles frees the evaluated scene once it has synced it (to save memory in final
    renders), so anything read after it runs must come from here."""
    from types import SimpleNamespace as NS
    insts = []
    for inst in depsgraph.object_instances:
        ob = inst.object
        M = inst.matrix_world.copy()
        if sd.is_splat_object(ob) and not inst.is_instance:
            original = bpy.data.objects.get(ob.name) or ob
            proxy = NS(name=ob.name, type=ob.type, data=original.data, matrix_world=ob.matrix_world.copy())
        elif ob.type == 'LIGHT':
            L = ob.data
            proxy = NS(name=ob.name, type='LIGHT',
                       data=NS(type=L.type, color=tuple(L.color), energy=float(L.energy),
                               spot_size=getattr(L, "spot_size", 0.0),
                               spot_blend=getattr(L, "spot_blend", 0.0)))
        else:
            continue
        insts.append(NS(object=proxy, matrix_world=M, is_instance=inst.is_instance))
    return NS(object_instances=insts)


def _frozen_settings(settings):
    from types import SimpleNamespace as NS
    return NS(**{p.identifier: getattr(settings, p.identifier)
                 for p in settings.bl_rna.properties if p.identifier != "rna_type"})


def _splat_objects(depsgraph):
    return [inst.object for inst in depsgraph.object_instances
            if not inst.is_instance and sd.is_splat_object(inst.object)]


def _render_view(depsgraph, settings, cam_world, V_of_world, fx, fy, cx, cy, W, H, report,
                 follow_camera=True, quick=False, viewport=False, mesh=None):
    """Render the scene's splat object. Returns linear premultiplied rgb, alpha, ray depth.

    mesh: optional (triangles, surface points, key): meshes that shadow the splats, and mesh
    points to find the splats' shadow at (read back from the relighter's mesh_vis)."""
    splats = _splat_objects(depsgraph)
    if not splats:
        return None
    obj = None
    with _LOCK:
        if mesh is not None:
            _MESH.update(tris=mesh[0], points=mesh[1], serial=mesh[2])
        try:
            _upload_scene(splats, report)
            _ensure_colours(depsgraph, obj, cam_world, settings, report, follow_camera, quick,
                            viewport)
        finally:
            _MESH.update(tris=None, points=None)
            if _CACHE.get("relighter") is not None:
                _CACHE["relighter"].mesh_tris = _CACHE["relighter"].mesh_points = None
        rgb, alpha, d = native.render(V_of_world(), fx, fy, cx, cy, W, H, settings.primitive)
    return _to_linear(rgb, alpha, settings), alpha, d


def _to_linear(rgb, alpha, settings):
    if settings.colour_input == 'SRGB':
        # The renderer blends in the model's (sRGB) space; Blender wants linear.
        a = alpha[..., None]
        straight = np.where(a > 1e-6, rgb / np.maximum(a, 1e-6), 0.0)
        rgb = sd.srgb_to_linear(straight) * a
    return rgb


def _render_mirror(n, off, cam_world, fx, fy, cx, cy, W, H, settings, lights=None):
    """The splats as seen in a flat mirror (plane n.x = off) from this camera, with their
    current colours: rendered from the camera reflected in the plane, leaving out splats on
    the far side of it. Returns linear premultiplied rgb, alpha."""
    d = _CACHE["splats"]
    dc, sh = _CACHE.get("gpu_colours", (d["dc"], d["sh"]))
    mdc, msh = dc, sh
    if settings.use_relight and _CACHE.get("relighter") is not None and lights is not None:
        # the shine and view colour were worked out for the camera, not for the mirror's view
        mdc, msh = _CACHE["relighter"].diffuse_dc(lights, settings.ambient, settings.exposure), None
    n = np.asarray(n, np.float64)
    C = np.array(cam_world, np.float64)
    side = 1.0 if n @ C[:3, 3] - off >= 0 else -1.0
    keep = side * (d["means"].astype(np.float64) @ n - off) > 0
    logit = np.where(keep, d["logit_opacities"], -30.0).astype(np.float32)
    Refl = np.eye(4)
    Refl[:3, :3] -= 2 * np.outer(n, n)
    Refl[:3, 3] = 2 * off * n
    V = np.array(sd.view_matrix(cam_world, _IDENTITY), np.float64)
    # the reflection is left-handed; flip the image's x to keep the camera a rotation
    Vm = (np.diag([-1.0, 1.0, 1.0, 1.0]) @ V @ Refl).astype(np.float32)
    with _LOCK:
        native.set_splats(d["means"], d["quats"], d["log_scales"], logit, d["dc"], d["sh"], d["degree"])
        native.set_colors(mdc, msh)
        if msh is None:
            native.set_render_degree(0)          # relit colours are base colour only
        try:
            rgb, alpha, _ = native.render(Vm, fx, fy, W - cx, cy, W, H, settings.primitive)
        finally:
            native.set_splats(d["means"], d["quats"], d["log_scales"], d["logit_opacities"],
                              d["dc"], d["sh"], d["degree"])
            native.set_colors(dc, sh)
    rgb, alpha = rgb[:, ::-1], alpha[:, ::-1]
    return _to_linear(rgb, alpha, settings), alpha


def _blur(img, sigma):
    """Approximate gaussian (three box passes) over the first two axes."""
    r = int(round(sigma * 0.85))
    if r < 1:
        return img
    k = 2 * r + 1
    out = img
    for _ in range(3):
        for axis in (0, 1):
            pad = [(0, 0)] * out.ndim
            pad[axis] = (r + 1, r)
            c = np.cumsum(np.pad(out, pad, mode='edge'), axis=axis)
            out = ((c[k:] - c[:-k]) if axis == 0 else (c[:, k:] - c[:, :-k])) / k
    return out.astype(np.float32)


def _write_pass(render_pass, pixels):
    """Bulk copy into a render pass (element-wise assignment takes seconds at HD)."""
    flat = np.ascontiguousarray(pixels, np.float32).ravel()
    rect = render_pass.rect
    if hasattr(rect, "foreach_set"):
        rect.foreach_set(flat)
    else:
        render_pass.rect = flat.reshape(-1, render_pass.channels)


def _viewport_camera(rv3d, W, H, obj_world, scene_size):
    """Viewport projection -> pinhole intrinsics and an object->camera matrix (OpenCV axes).

    Orthographic views are approximated by a far-away camera with a matching narrow lens."""
    P = np.array(rv3d.window_matrix, np.float64)
    Vw = np.array(rv3d.view_matrix, np.float64)            # world -> view (OpenGL axes)
    if rv3d.is_perspective:
        fx, fy = P[0, 0] * W / 2, P[1, 1] * H / 2
        cx, cy = W / 2 * (1 - P[0, 2]), H / 2 * (1 + P[1, 2])
        back = 0.0
    else:
        back = scene_size * 200.0
        fx, fy = P[0, 0] * W / 2 * back, P[1, 1] * H / 2 * back
        cx, cy = W / 2 * (1 - P[0, 3]), H / 2 * (1 + P[1, 3])
        Vw = Vw.copy()
        Vw[2, 3] -= back                                  # step the eye back along the view
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    V = (flip @ Vw @ np.array(obj_world, np.float64)).astype(np.float32)
    cam_world = np.linalg.inv(Vw)
    return V, fx, fy, cx, cy, cam_world


REFINE_DELAY = 0.5      # seconds the lights must be still before the full relight


_REFINE = {"changed_at": 0.0, "pending": False}


def _schedule_refine():
    """Redraw the 3D views once the lights have been still for REFINE_DELAY, so the full
    relight replaces the quick one."""
    if _REFINE["pending"]:
        return
    _REFINE["pending"] = True

    def check():
        if time.time() - _REFINE["changed_at"] < REFINE_DELAY:
            return 0.1                           # still moving: look again shortly
        _REFINE["pending"] = False
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        return None
    bpy.app.timers.register(check, first_interval=REFINE_DELAY)


class SpirulaRenderEngine(bpy.types.RenderEngine):
    bl_idname = "SPIRULA"
    bl_label = "Spirula (3DGS)"
    bl_use_preview = False
    bl_use_postprocess = True
    bl_use_eevee_viewport = False

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.session = None
        self._view = {"key": None, "texture": None, "size": None, "dirty": True,
                      "lights": None, "quality": None, "changed_at": 0.0}

    # ------------------------------------------------------------ final render
    def __del__(self):
        try:
            self._free_cycles()
        except ReferenceError:
            pass

    def _free_cycles(self):
        if getattr(self, "session", None):
            import cycles.engine as ce
            ce.free(self)
        self.session = None

    def update(self, data, depsgraph):
        """Final renders with polygon objects: start a Cycles session that renders them into
        this engine's result (meshes.prepare_cycles has set the scene up for it)."""
        self.session = None
        if not meshes.prepared():
            return
        import cycles.engine as ce
        ce.create(self, data)
        ce.reset(self, data, depsgraph)

    def update_render_passes(self, scene=None, renderlayer=None):
        self.register_pass(scene, renderlayer, "Combined", 4, "RGBA", 'COLOR')
        if renderlayer.use_pass_z:
            self.register_pass(scene, renderlayer, "Depth", 1, "Z", 'VALUE')
        if meshes.prepared():
            import cycles.engine as ce
            ce.register_passes(self, scene, renderlayer)

    def _render_meshes(self, depsgraph):
        """Cycles renders the polygon objects; returns their passes, or None."""
        if not getattr(self, "session", None):
            return None
        import cycles.engine as ce
        t = time.time()
        self.update_stats("", "Spirula: path tracing the meshes (Cycles)")
        ce.render(self, depsgraph)
        P = meshes.read_passes(self.get_result())
        self._free_cycles()
        print(f"Spirula: meshes (Cycles) in {time.time() - t:.2f}s")
        return P

    def render(self, depsgraph):
        scene = depsgraph.scene
        settings = _frozen_settings(scene.spirula)
        cam_obj = scene.camera
        if cam_obj is None:
            self.report({'ERROR'}, "The scene has no camera")
            return
        try:
            W, H, fx, fy, cx, cy = sd.camera_intrinsics(scene, cam_obj)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return
        cam_world = cam_obj.evaluated_get(depsgraph).matrix_world.copy()
        film_transparent = meshes.original_film_transparent(scene)
        world = tuple(scene.world.color)[:3] if scene.world else (0.05, 0.05, 0.05)
        frozen = _freeze(depsgraph)
        tris = None
        mirrors = []
        if getattr(self, "session", None):
            if settings.use_shadows:
                tris = meshes.world_triangles(depsgraph)
            if settings.use_reflections:
                mirrors = meshes.mirror_planes(depsgraph)
        P = self._render_meshes(depsgraph)            # the depsgraph is gone after this
        if self.test_break():
            return
        ys, xs = np.mgrid[0:H, 0:W]
        ray = np.sqrt(1 + ((xs + 0.5 - cx) / fx) ** 2 + ((ys + 0.5 - cy) / fy) ** 2)
        mesh = None
        if P is not None:
            mesh = self._mesh_layer(P, tris, settings, ray)
        t0 = time.time()
        self.update_stats("", "Spirula: rendering")
        out = _render_view(frozen, settings, cam_world,
                           lambda: sd.view_matrix(cam_world, _IDENTITY),
                           fx, fy, cx, cy, W, H, lambda m: self.report({'WARNING'}, m),
                           quick=False,
                           mesh=None if mesh is None else (mesh["tris"], mesh["points"],
                                                           ("final", next(_SERIAL))))
        mesh_vis = None
        if mesh is not None and _CACHE.get("relighter") is not None:
            mesh_vis = _CACHE["relighter"].mesh_vis
        if out is None:
            rgb = np.zeros((H, W, 3), np.float32)
            alpha = np.zeros((H, W), np.float32)
            depth = np.full((H, W), np.inf, np.float32)
        else:
            rgb, alpha, d = out
            depth = np.where(alpha > 1e-3, d, np.inf).astype(np.float32)
            print(f"Spirula: {W}x{H} in {time.time() - t0:.2f}s")

        if mesh is not None and out is not None and mirrors:
            mesh["reflection"] = self._reflections(mesh, mirrors, cam_world, fx, fy, cx, cy, W, H,
                                                   settings, _object_space_lights(frozen, _IDENTITY))
        rgba = np.concatenate([rgb, alpha[..., None]], -1)
        if mesh is not None:
            rgba, depth = self._composite(rgba, depth, mesh, mesh_vis, settings)
        if not film_transparent:
            bg = np.array(world, np.float32)
            rgba[..., :3] += bg * (1 - rgba[..., 3:4])
            rgba[..., 3] = 1.0
        rgba = np.ascontiguousarray(rgba[::-1])            # Blender rows run bottom-up
        result = self.begin_result(0, 0, W, H)
        layer = result.layers[0]
        _write_pass(layer.passes["Combined"], rgba)
        zp = layer.passes.get("Depth")
        if zp is not None:
            # ray distance -> distance along the view axis, as Blender's Z pass stores
            z = depth / ray
            z = np.where(np.isfinite(z), z, 1e10).astype(np.float32)[::-1]
            _write_pass(zp, z)
        self.end_result(result)

    def _mesh_layer(self, P, tris, settings, ray):
        """Cycles' passes -> what the composite needs; also hands the relighter the mesh
        triangles (shadows on the splats) and surface points (splat shadows on the meshes)."""
        M = P["Combined"]
        a = M[..., 3]
        covered = a > 1e-4
        pos = P["Position"] / np.maximum(a, 1e-6)[..., None]       # undo edge coverage
        z = P["Depth"][..., 0]
        dist = np.where(covered & (z < 1e9), z * ray, np.inf).astype(np.float32)
        groups = {}
        for name, pass_name in meshes.light_groups().items():
            if pass_name in P:
                groups[name] = P[pass_name][..., :3]
        noisy = P.get("Noisy Image", M)[..., :3]          # light groups aren't denoised
        lights_sum = sum(groups.values()) if groups else 0.0
        points = np.ascontiguousarray(pos[covered], np.float32) \
            if settings.use_shadows and settings.use_relight else None
        glossy = P.get("Glossy Color")
        index = P.get("IndexOB", P.get("Object Index"))
        on_top = None if index is None else np.abs(index[..., 0] - meshes.ON_TOP_INDEX) < 0.5
        return dict(rgba=M, covered=covered, dist=dist, groups=groups, tris=tris, points=points,
                    pos=pos.astype(np.float32), glossy=None if glossy is None else glossy[..., :3],
                    on_top=on_top,
                    noisy=noisy, rest=noisy - lights_sum, albedo=P.get("Diffuse Color"))

    def _reflections(self, mesh, mirrors, cam_world, fx, fy, cx, cy, W, H, settings, lights):
        """The splats reflected in flat shiny meshes: per pixel, Cycles' glossy colour (how
        much that surface reflects, Fresnel included) times the mirrored splat render,
        blurred by the surface's roughness."""
        gc = mesh.get("glossy")
        if gc is None:
            return None
        pos, cov = mesh["pos"], mesh["covered"]
        extent = _CACHE["relighter"].extent if _CACHE.get("relighter") is not None else 1.0
        total = np.zeros((H, W, 3), np.float32)
        for n, off, rough, name in mirrors:
            on = cov & (np.abs(pos @ np.asarray(n, np.float32) - off) < 1e-4 * max(extent, 1e-3) + 1e-6)
            on &= gc.max(-1) > 1e-3
            if on.sum() < 16:
                continue
            t = time.time()
            self.update_stats("", f"Spirula: reflection in {name}")
            rgb, _ = _render_mirror(n, off, cam_world, fx, fy, cx, cy, W, H, settings, lights)
            rgb = _blur(rgb, (1.0 + 60.0 * rough * rough) * W / 1920.0)
            total[on] += gc[on] * rgb[on]
            print(f"Spirula: reflection in '{name}' in {time.time() - t:.2f}s")
        return total

    def _composite(self, rgba, depth, mesh, mesh_vis, settings):
        """Splats and meshes, per pixel by depth. Meshes are relit in this engine's units with
        each light dimmed by the splats' shadow."""
        M = mesh["rgba"].copy()
        if settings.use_relight and mesh["groups"]:
            gain = 2.0 ** settings.exposure
            lit = mesh["rest"].copy()
            cov = mesh["covered"]
            for name, g in mesh["groups"].items():
                k = np.float32(meshes.CYCLES_UNIT * gain)
                if mesh_vis and name in mesh_vis:
                    vis = np.ones(g.shape[:2], np.float32)
                    vis[cov] = mesh_vis[name]
                    lit += k * g * vis[..., None]
                else:
                    lit += k * g
            # Light groups and the rest come undenoised: apply the change they make as a
            # ratio to Cycles' (possibly denoised) image, so its noise doesn't come back.
            eps = np.float32(1e-4)
            ratio = (lit + eps) / (mesh["noisy"] + eps)
            lit = M[..., :3] * ratio
            if mesh["albedo"] is not None:
                lit += np.float32(settings.ambient) * mesh["albedo"][..., :3]
            M[..., :3] = lit
        if mesh.get("reflection") is not None:
            M[..., :3] += mesh["reflection"]
        S = rgba
        aS, aM = S[..., 3:4], M[..., 3:4]
        s_over = S + (1 - aS) * M
        m_over = M + (1 - aM) * S
        extent = _CACHE["relighter"].extent if _CACHE.get("relighter") is not None else 1.0
        dS = np.where(np.isfinite(depth), depth, 1e30)
        dM = np.where(np.isfinite(mesh["dist"]), mesh["dist"], 1e30)
        # soft over a hair's width, so contact (feet on a floor) doesn't alias
        w = 1 / (1 + np.exp(np.clip((dS - dM) / (0.002 * extent), -40, 40)))
        if mesh.get("on_top") is not None:
            w = np.where(mesh["on_top"], 0.0, w)            # labels, rulers: over the splats
        w = w[..., None].astype(np.float32)
        out = w * s_over + (1 - w) * m_over
        depth = np.minimum(depth, mesh["dist"])
        return out.astype(np.float32), depth

    # ------------------------------------------------------------ viewport
    def view_update(self, context, depsgraph):
        # Something in the scene changed (an edit, a light, a setting): redraw fresh.
        self._view["dirty"] = True
        self._view["mesh_dirty"] = True

    def _viewport_mesh_layer(self, depsgraph):
        v = self._view
        layer = v.get("mesh_layer")
        if layer is None:
            layer = v["mesh_layer"] = viewport_meshes.MeshLayer()
            v["mesh_dirty"] = True
        if v.get("mesh_dirty", True):
            layer.rebuild(depsgraph)
            v["mesh_dirty"] = False
        return layer

    def view_draw(self, context, depsgraph):
        region, rv3d, scene = context.region, context.region_data, depsgraph.scene
        settings = scene.spirula
        scale = {'FULL': 1, 'HALF': 2, 'QUARTER': 4}[settings.viewport_resolution]
        if settings.use_relight and time.time() - _REFINE["changed_at"] < REFINE_DELAY:
            scale *= 2                               # lights moving: a lighter preview
        W, H = max(1, region.width // scale), max(1, region.height // scale)
        view_key = (tuple(map(tuple, rv3d.perspective_matrix)), W, H)
        v = self._view
        mesh_on = settings.use_meshes and meshes.has_meshes(depsgraph)
        mesh_key = meshes.instances_key(depsgraph) if mesh_on else None
        # While the lights (or the meshes shadowing the splats) keep changing, relight the
        # quick way; once they've been still for a moment, the timer asks for the full relight.
        quick = False
        if settings.use_relight:
            splats_now = _splat_objects(depsgraph)
            if splats_now:
                # Relit colours depend on the lights AND (for shine and view-dependent
                # colour) on where the camera is, so moving either refines afterwards.
                lk = (_lights_key(_object_space_lights(depsgraph, _IDENTITY),
                                  settings, settings.use_shadows), view_key[0], mesh_key)
                now = time.time()
                if lk != v["lights"]:
                    if v["lights"] is not None:
                        _REFINE["changed_at"] = now
                    v["lights"] = lk
                quick = now - _REFINE["changed_at"] < REFINE_DELAY
                if quick:
                    _schedule_refine()
                if v["quality"] != ("quick" if quick else "full"):
                    v["dirty"] = True
        if v["dirty"] or v["key"] != view_key or v["texture"] is None or \
                v.get("mesh_mode") != mesh_on:
            v["quality"] = "quick" if quick else "full"
            v["dirty"] = False
            v["key"] = view_key
            v["mesh_mode"] = mesh_on
            splats = _splat_objects(depsgraph)
            size = 1.0
            if splats and _CACHE.get("relighter") is not None:
                size = _CACHE["relighter"].extent
            obj_world = _IDENTITY
            V, fx, fy, cx, cy, cam_world = _viewport_camera(rv3d, W, H, obj_world, size)
            mesh = None
            covered = None
            if mesh_on and settings.use_relight and settings.use_shadows:
                tris = meshes.world_triangles(depsgraph)
                points = None
                if settings.viewport_shadows:
                    # where the meshes are on screen (at the splat render's size), for the
                    # splats' shadow on them
                    layer = self._viewport_mesh_layer(depsgraph)
                    layer.draw(rv3d, region.width, region.height, [], 1.0, 1.0)
                    pos = layer.read_positions(region.width, region.height)
                    pos = pos[::-1][:H * scale:scale, :W * scale:scale]
                    covered = pos[..., 3] > 0.5
                    points = np.ascontiguousarray(pos[covered][:, :3], np.float32)
                    if not len(points):
                        points, covered = None, None
                mesh = (tris, points, ("viewport", mesh_key, view_key[0], next(_SERIAL)))
            try:
                out = _render_view(depsgraph, settings, cam_world, lambda: V, fx, fy, cx, cy,
                                   W, H, lambda m: None, follow_camera=not quick, quick=quick,
                                   viewport=True, mesh=mesh)
            except native.RenderError as e:
                out = None
                print("Spirula viewport:", e)
            v["vis"] = None
            mesh_vis = _CACHE["relighter"].mesh_vis if (covered is not None and
                                                        _CACHE.get("relighter") is not None) else None
            if mesh_vis:
                v["vis"] = []
                for L in _object_space_lights(depsgraph, _IDENTITY)[:viewport_meshes.SHADOWED_LIGHTS]:
                    img = np.ones((H, W), np.float32)
                    if L["name"] in mesh_vis:
                        img[covered] = mesh_vis[L["name"]]
                    img = np.ascontiguousarray(img[::-1])
                    v["vis"].append(gpu.types.GPUTexture(
                        (W, H), format='R32F', data=gpu.types.Buffer('FLOAT', W * H, img.ravel())))
            if out is None:
                rgba = np.zeros((H, W, 4), np.float32)
                d = np.full((H, W), np.inf, np.float32)
            else:
                rgb, alpha, d = out
                rgba = np.concatenate([rgb, alpha[..., None]], -1)
            world = scene.world.color if scene.world else (0.05, 0.05, 0.05)
            bg = np.array(list(world)[:3], np.float32)
            if not mesh_on:   # the viewport always shows the world colour behind the splats
                rgba[..., :3] += bg * (1 - rgba[..., 3:4])
                rgba[..., 3] = 1.0
            else:
                # splat depth along the view axis, as the mesh pass writes it
                ys, xs = np.mgrid[0:H, 0:W]
                ray = np.sqrt(1 + ((xs + 0.5 - cx) / fx) ** 2 + ((ys + 0.5 - cy) / fy) ** 2)
                back = 0.0 if rv3d.is_perspective else size * 200.0
                z = np.where(rgba[..., 3] > 1e-3, d / ray - back, 1e30).astype(np.float32)
                z = np.ascontiguousarray(z[::-1])
                v["ztex"] = gpu.types.GPUTexture((W, H), format='R32F',
                                                 data=gpu.types.Buffer('FLOAT', W * H, z.ravel()))
            rgba = np.ascontiguousarray(rgba[::-1], np.float32)
            buf = gpu.types.Buffer('FLOAT', W * H * 4, rgba.ravel())
            v["texture"] = gpu.types.GPUTexture((W, H), format='RGBA16F', data=buf)
        tex = v["texture"]
        if mesh_on:
            layer = self._viewport_mesh_layer(depsgraph)
            lights = _object_space_lights(depsgraph, _IDENTITY) if settings.use_relight else []
            ambient = settings.ambient if settings.use_relight else 1.0
            layer.draw(rv3d, region.width, region.height, lights, ambient, 2.0 ** settings.exposure,
                       v.get("vis"))
            world = scene.world.color if scene.world else (0.05, 0.05, 0.05)
            extent = _CACHE["relighter"].extent if _CACHE.get("relighter") is not None else 1.0
            tex = layer.composite(v["texture"], v["ztex"], tuple(world)[:3], 0.002 * extent,
                                  region.width, region.height)
        gpu.state.blend_set('ALPHA_PREMULT')
        self.bind_display_space_shader(scene)
        draw_texture_2d(tex, (0, 0), region.width, region.height)
        self.unbind_display_space_shader()
        gpu.state.blend_set('NONE')
