"""The Spirula render engine: final renders (F12, animation) and the viewport's Rendered mode."""

import math
import threading
import time

import bpy
import gpu
import numpy as np
from gpu_extras.presets import draw_texture_2d

from . import native, relight, scene_data as sd

# One renderer (one GPU context and one uploaded splat set) serves the viewport and
# final renders, which can run at the same time on different threads.
_LOCK = threading.RLock()
_CACHE = {"key": None, "colour_key": None}


def _splat_key(obj):
    pc = obj.data
    return (pc.session_uid, len(pc.points), int(pc.get("gsp_revision", 0)),
            obj.name, tuple(a.name for a in pc.attributes))


def _upload(obj):
    key = _splat_key(obj)
    if _CACHE["key"] == key:
        return False
    d = sd.splat_arrays(obj.data)
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
            shadows, settings.shadow_resolution, settings.shadow_softness)


def _ensure_colours(depsgraph, obj, cam_world, settings, report, follow_camera=True):
    """Put the right colours on the GPU: the original ones, or relit by the scene's lights.

    follow_camera=False (viewport) keeps the last relit colours while only the view
    moves; lighting is redone when lights or settings change."""
    d = _CACHE["splats"]
    if not settings.use_relight:
        if _CACHE["colour_key"] is not None:
            native.set_colors(d["dc"], d["sh"])
            _CACHE["colour_key"] = None
        return
    if _CACHE.get("relighter") is None:
        _CACHE["relighter"] = relight.Relighter(d)
    lights = _object_space_lights(depsgraph, obj.matrix_world)
    Minv = np.linalg.inv(np.array(obj.matrix_world, np.float64))
    cam_pos = (Minv @ np.array(cam_world, np.float64)[:, 3])[:3].astype(np.float32)
    shadows = settings.use_shadows and (follow_camera or settings.viewport_shadows)
    key = _lights_key(lights, settings, shadows)
    if follow_camera:
        key = key + (tuple(np.round(cam_pos, 5)),)
    if _CACHE["colour_key"] is not None and _CACHE["colour_key"][:len(key)] == key:
        return
    dc, sh = _CACHE["relighter"].colours(
        lights, cam_pos, settings.ambient, settings.exposure, shadows,
        settings.shadow_resolution, settings.shadow_softness, settings.primitive, report)
    native.set_colors(dc, sh)
    _CACHE["colour_key"] = key


def _splat_objects(depsgraph):
    return [inst.object for inst in depsgraph.object_instances
            if not inst.is_instance and sd.is_splat_object(inst.object)]


def _render_view(depsgraph, settings, cam_world, V_of_obj, fx, fy, cx, cy, W, H, report,
                 follow_camera=True):
    """Render the scene's splat object. Returns linear premultiplied rgb, alpha, ray depth."""
    splats = _splat_objects(depsgraph)
    if not splats:
        return None
    if len(splats) > 1:
        report(f"Spirula renders one splat object; using '{splats[0].name}' (join the others into it)")
    obj = splats[0]
    original = bpy.data.objects.get(obj.name)              # attribute data lives here
    with _LOCK:
        _upload(original if original is not None else obj)
        _ensure_colours(depsgraph, obj, cam_world, settings, report, follow_camera)
        rgb, alpha, d = native.render(V_of_obj(obj), fx, fy, cx, cy, W, H, settings.primitive)
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


class SpirulaRenderEngine(bpy.types.RenderEngine):
    bl_idname = "SPIRULA"
    bl_label = "Spirula (3DGS)"
    bl_use_preview = False
    bl_use_postprocess = True
    bl_use_eevee_viewport = False

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._view = {"key": None, "texture": None, "size": None, "dirty": True}

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
                           lambda obj: sd.view_matrix(cam_world, obj.matrix_world),
                           fx, fy, cx, cy, W, H, lambda m: self.report({'WARNING'}, m))
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
        W, H = max(1, region.width // scale), max(1, region.height // scale)
        view_key = (tuple(map(tuple, rv3d.perspective_matrix)), W, H)
        v = self._view
        if v["dirty"] or v["key"] != view_key or v["texture"] is None:
            v["dirty"] = False
            v["key"] = view_key
            splats = _splat_objects(depsgraph)
            size = 1.0
            if splats and _CACHE.get("relighter") is not None:
                size = _CACHE["relighter"].extent
            obj_world = splats[0].matrix_world if splats else np.eye(4)
            V, fx, fy, cx, cy, cam_world = _viewport_camera(rv3d, W, H, obj_world, size)
            try:
                out = _render_view(depsgraph, settings, cam_world, lambda obj: V, fx, fy, cx, cy,
                                   W, H, lambda m: None, follow_camera=False)
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
