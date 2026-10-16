"""The Spirula render engine: final renders (F12, animation) of 3DGS objects."""

import time

import bpy
import numpy as np

import math

from . import native, relight, scene_data as sd

# What's uploaded right now, so animation frames don't re-upload unchanged splats.
_CACHE = {"key": None}


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
    _CACHE.update(key=key, n=d["n"], splats=d, relighter=None, lit=False)
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


def _apply_lighting(depsgraph, obj, cam_world, settings, report):
    """Relight (or restore) the uploaded splats' colours for this frame."""
    d = _CACHE["splats"]
    if not settings.use_relight:
        if _CACHE.get("lit"):
            native.set_colors(d["dc"], d["sh"])
            _CACHE["lit"] = False
        return
    if _CACHE.get("relighter") is None:
        _CACHE["relighter"] = relight.Relighter(d)
    lights = _object_space_lights(depsgraph, obj.matrix_world)
    Minv = np.linalg.inv(np.array(obj.matrix_world, np.float64))
    cam_pos = (Minv @ np.array(cam_world, np.float64)[:, 3])[:3].astype(np.float32)
    dc, sh = _CACHE["relighter"].colours(
        lights, cam_pos, settings.ambient, settings.exposure, settings.use_shadows,
        settings.shadow_resolution, settings.shadow_softness, settings.primitive, report)
    native.set_colors(dc, sh)
    _CACHE["lit"] = True


def _write_pass(render_pass, pixels):
    """Bulk copy into a render pass (element-wise assignment takes seconds at HD)."""
    flat = np.ascontiguousarray(pixels, np.float32).ravel()
    rect = render_pass.rect
    if hasattr(rect, "foreach_set"):
        rect.foreach_set(flat)
    else:
        render_pass.rect = flat.reshape(-1, render_pass.channels)


class SpirulaRenderEngine(bpy.types.RenderEngine):
    bl_idname = "SPIRULA"
    bl_label = "Spirula (3DGS)"
    bl_use_preview = False
    bl_use_postprocess = True

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

        # The render depsgraph only holds objects enabled for rendering.
        splats = [inst.object for inst in depsgraph.object_instances
                  if not inst.is_instance and sd.is_splat_object(inst.object)]
        rgb = np.zeros((H, W, 3), np.float32)
        alpha = np.zeros((H, W), np.float32)
        depth = np.full((H, W), np.inf, np.float32)
        if len(splats) > 1:
            self.report({'WARNING'}, "Spirula renders one splat object; using "
                        f"'{splats[0].name}' (join the others into it)")
        if splats:
            obj = splats[0]
            original = bpy.data.objects.get(obj.name)          # attribute data lives here
            t0 = time.time()
            self.update_stats("", f"Spirula: loading {obj.name}")
            uploaded = _upload(original if original is not None else obj)
            if settings.use_relight or _CACHE.get("lit"):
                self.update_stats("", "Spirula: relighting")
                _apply_lighting(depsgraph, obj, cam_world, settings,
                                lambda m: self.report({'WARNING'}, m))
            V = sd.view_matrix(cam_world, obj.matrix_world)
            self.update_stats("", f"Spirula: rendering {_CACHE['n']:,} splats")
            rgb, alpha, d = native.render(V, fx, fy, cx, cy, W, H, settings.primitive)
            if settings.colour_input == 'SRGB':
                # The renderer blends in the model's (sRGB) space; Blender wants linear.
                a = alpha[..., None]
                straight = np.where(a > 1e-6, rgb / np.maximum(a, 1e-6), 0.0)
                rgb = sd.srgb_to_linear(straight) * a
            depth = np.where(alpha > 1e-3, d, np.inf).astype(np.float32)
            print(f"Spirula: {obj.name} {W}x{H} in {time.time() - t0:.2f}s"
                  f"{' (uploaded splats)' if uploaded else ''}")

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
