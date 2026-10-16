"""The Spirula render engine: final renders (F12, animation) and the viewport's Rendered mode."""

import math
import threading
import time

import bpy
import gpu
import numpy as np
from gpu_extras.presets import draw_texture_2d

from . import native, relight, scene_data as sd, transform

# One renderer (one GPU context and one uploaded splat set) serves the viewport and
# final renders, which can run at the same time on different threads.
_LOCK = threading.RLock()
_CACHE = {"key": None, "colour_key": None}


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
            _CACHE["colour_key"] = None
        native.set_render_degree(-1)
        return None
    if _CACHE.get("relighter") is None:
        _CACHE["relighter"] = relight.Relighter(d)
    _CACHE["relighter"].shine_colour = settings.shine_colour
    _CACHE["relighter"].shadow_density = settings.shadow_density
    lights = _object_space_lights(depsgraph, _IDENTITY)
    Minv = _IDENTITY
    cam_pos = (Minv @ np.array(cam_world, np.float64)[:, 3])[:3].astype(np.float32)
    shadows = settings.use_shadows and (not viewport or settings.viewport_shadows) and not quick
    key = _lights_key(lights, settings, shadows) + ("quick" if quick else "full",)
    if follow_camera:
        key = key + (tuple(np.round(cam_pos, 5)),)
    if _CACHE["colour_key"] != key and quick:
        native.set_colors(_CACHE["relighter"].quick_dc(lights, cam_pos, settings.ambient,
                                                       settings.exposure), None)
        _CACHE["colour_key"] = key
    elif _CACHE["colour_key"] != key:
        dc, sh = _CACHE["relighter"].colours(
            lights, cam_pos, settings.ambient, settings.exposure, shadows,
            settings.shadow_resolution, settings.shadow_softness, settings.primitive, report,
            diffuse_only=quick)
        native.set_colors(dc, sh)
        _CACHE["colour_key"] = key
    # Relit colours are baked per splat for this camera: render without the higher bands.
    native.set_render_degree(0)
    return _lights_key(lights, settings, settings.use_shadows)


def _splat_objects(depsgraph):
    return [inst.object for inst in depsgraph.object_instances
            if not inst.is_instance and sd.is_splat_object(inst.object)]


def _render_view(depsgraph, settings, cam_world, V_of_world, fx, fy, cx, cy, W, H, report,
                 follow_camera=True, quick=False, viewport=False):
    """Render the scene's splat object. Returns linear premultiplied rgb, alpha, ray depth."""
    splats = _splat_objects(depsgraph)
    if not splats:
        return None
    obj = None
    with _LOCK:
        _upload_scene(splats, report)
        _ensure_colours(depsgraph, obj, cam_world, settings, report, follow_camera, quick, viewport)
        rgb, alpha, d = native.render(V_of_world(), fx, fy, cx, cy, W, H, settings.primitive)
    if settings.colour_input == 'SRGB':
        # The renderer blends in the model's (sRGB) space; Blender wants linear.
        a = alpha[..., None]
        straight = np.where(a > 1e-6, rgb / np.maximum(a, 1e-6), 0.0)
        rgb = sd.srgb_to_linear(straight) * a
    return rgb, alpha, d


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
        self._view = {"key": None, "texture": None, "size": None, "dirty": True,
                      "lights": None, "quality": None, "changed_at": 0.0}

    # ------------------------------------------------------------ final render
    def update_render_passes(self, scene=None, renderlayer=None):
        self.register_pass(scene, renderlayer, "Combined", 4, "RGBA", 'COLOR')
        if renderlayer.use_pass_z:
            self.register_pass(scene, renderlayer, "Depth", 1, "Z", 'VALUE')

    def render(self, depsgraph):
        scene = depsgraph.scene
        settings = scene.spirula
        cam_obj = scene.camera
        if cam_obj is None:
            self.report({'ERROR'}, "The scene has no camera")
            return
        try:
            W, H, fx, fy, cx, cy = sd.camera_intrinsics(scene, cam_obj)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return
        cam_world = cam_obj.evaluated_get(depsgraph).matrix_world
        t0 = time.time()
        self.update_stats("", "Spirula: rendering")
        out = _render_view(depsgraph, settings, cam_world,
                           lambda: sd.view_matrix(cam_world, _IDENTITY),
                           fx, fy, cx, cy, W, H, lambda m: self.report({'WARNING'}, m),
                           quick=False)
        if out is None:
            rgb = np.zeros((H, W, 3), np.float32)
            alpha = np.zeros((H, W), np.float32)
            depth = np.full((H, W), np.inf, np.float32)
        else:
            rgb, alpha, d = out
            depth = np.where(alpha > 1e-3, d, np.inf).astype(np.float32)
            print(f"Spirula: {W}x{H} in {time.time() - t0:.2f}s")

        rgba = np.concatenate([rgb, alpha[..., None]], -1)
        if not scene.render.film_transparent:
            world = scene.world.color if scene.world else (0.05, 0.05, 0.05)
            bg = np.array(list(world)[:3], np.float32)
            rgba[..., :3] += bg * (1 - rgba[..., 3:4])
            rgba[..., 3] = 1.0
        rgba = np.ascontiguousarray(rgba[::-1])            # Blender rows run bottom-up
        result = self.begin_result(0, 0, W, H)
        layer = result.layers[0]
        _write_pass(layer.passes["Combined"], rgba)
        zp = layer.passes.get("Depth")
        if zp is not None:
            # ray distance -> distance along the view axis, as Blender's Z pass stores
            ys, xs = np.mgrid[0:H, 0:W]
            dx, dy = (xs + 0.5 - cx) / fx, (ys + 0.5 - cy) / fy
            z = depth / np.sqrt(1 + dx * dx + dy * dy)
            z = np.where(np.isfinite(z), z, 1e10).astype(np.float32)[::-1]
            _write_pass(zp, z)
        self.end_result(result)

    # ------------------------------------------------------------ viewport
    def view_update(self, context, depsgraph):
        # Something in the scene changed (an edit, a light, a setting): redraw fresh.
        self._view["dirty"] = True

    def view_draw(self, context, depsgraph):
        region, rv3d, scene = context.region, context.region_data, depsgraph.scene
        settings = scene.spirula
        scale = {'FULL': 1, 'HALF': 2, 'QUARTER': 4}[settings.viewport_resolution]
        if settings.use_relight and time.time() - _REFINE["changed_at"] < REFINE_DELAY:
            scale *= 2                               # lights moving: a lighter preview
        W, H = max(1, region.width // scale), max(1, region.height // scale)
        view_key = (tuple(map(tuple, rv3d.perspective_matrix)), W, H)
        v = self._view
        # While the lights keep changing, relight the quick way; once they've been
        # still for a moment, the timer below asks for the full SH relight.
        quick = False
        if settings.use_relight:
            splats_now = _splat_objects(depsgraph)
            if splats_now:
                # Relit colours depend on the lights AND (for shine and view-dependent
                # colour) on where the camera is, so moving either refines afterwards.
                lk = (_lights_key(_object_space_lights(depsgraph, _IDENTITY),
                                  settings, settings.use_shadows), view_key[0])
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
        if v["dirty"] or v["key"] != view_key or v["texture"] is None:
            v["quality"] = "quick" if quick else "full"
            v["dirty"] = False
            v["key"] = view_key
            splats = _splat_objects(depsgraph)
            size = 1.0
            if splats and _CACHE.get("relighter") is not None:
                size = _CACHE["relighter"].extent
            obj_world = _IDENTITY
            V, fx, fy, cx, cy, cam_world = _viewport_camera(rv3d, W, H, obj_world, size)
            try:
                out = _render_view(depsgraph, settings, cam_world, lambda: V, fx, fy, cx, cy,
                                   W, H, lambda m: None, follow_camera=not quick, quick=quick,
                                   viewport=True)
            except native.RenderError as e:
                out = None
                print("Spirula viewport:", e)
            if out is None:
                rgba = np.zeros((H, W, 4), np.float32)
            else:
                rgb, alpha, _ = out
                rgba = np.concatenate([rgb, alpha[..., None]], -1)
            if True:   # the viewport always shows the world colour behind the splats
                world = scene.world.color if scene.world else (0.05, 0.05, 0.05)
                bg = np.array(list(world)[:3], np.float32)
                rgba[..., :3] += bg * (1 - rgba[..., 3:4])
                rgba[..., 3] = 1.0
            rgba = np.ascontiguousarray(rgba[::-1], np.float32)
            buf = gpu.types.Buffer('FLOAT', W * H * 4, rgba.ravel())
            v["texture"] = gpu.types.GPUTexture((W, H), format='RGBA16F', data=buf)
        gpu.state.blend_set('ALPHA_PREMULT')
        self.bind_display_space_shader(scene)
        draw_texture_2d(v["texture"], (0, 0), region.width, region.height)
        self.unbind_display_space_shader()
        gpu.state.blend_set('NONE')
