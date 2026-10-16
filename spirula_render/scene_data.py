"""Turning Blender data (splat point clouds, cameras) into the renderer's inputs."""

import math

import numpy as np

# Blender 5.3's 3DGS point-cloud attributes (see Blender's PLY importer).
POS, SCALE, ROT, BASE = "position", "scale", "rotation", "radiance:base"
HIDDEN = "gsp_hidden"          # erased by the Gaussian Splat Patch add-on

_KEY = {'FLOAT': 'value', 'BOOLEAN': 'value', 'FLOAT_VECTOR': 'vector', 'FLOAT4': 'vector',
        'QUATERNION': 'value'}
_WIDTH = {'FLOAT': 1, 'BOOLEAN': 1, 'FLOAT_VECTOR': 3, 'FLOAT4': 4, 'QUATERNION': 4}


def is_splat_object(obj):
    return (obj is not None and obj.type == 'POINTCLOUD'
            and obj.data.attributes.get(BASE) is not None
            and obj.data.attributes.get(ROT) is not None)


def _read(pc, name):
    a = pc.attributes.get(name)
    if a is None or a.domain != 'POINT' or a.data_type not in _KEY:
        return None
    n = len(pc.points)
    w = _WIDTH[a.data_type]
    buf = np.empty(n * w, bool if a.data_type == 'BOOLEAN' else np.float32)
    if n:
        a.data.foreach_get(_KEY[a.data_type], buf)
    return buf.reshape(n, w)


def splat_arrays(pc):
    """Renderer inputs for one point cloud, in its object space."""
    pos = _read(pc, POS)
    base = _read(pc, BASE)
    keep = None
    hidden = _read(pc, HIDDEN)
    if hidden is not None and hidden.any():
        keep = ~hidden[:, 0]
    sh_names = sorted((a.name for a in pc.attributes if a.name.startswith("radiance:sh_")),
                      key=lambda s: int(s.rsplit("_", 1)[1]))
    K = len(sh_names)
    degree = int(round(math.sqrt(K + 1))) - 1
    K = (degree + 1) ** 2 - 1                      # ignore any incomplete band
    sh = np.stack([_read(pc, nm) for nm in sh_names[:K]], 1) if K else None
    scale = _read(pc, SCALE)
    rot = _read(pc, ROT)
    op = np.clip(base[:, 3], 1e-6, 1 - 1e-6)
    out = dict(means=pos, quats=rot, log_scales=np.log(np.maximum(scale, 1e-12)),
               logit_opacities=np.log(op / (1 - op)), dc=base[:, :3], sh=sh)
    if keep is not None:
        out = {k: (v[keep] if v is not None else None) for k, v in out.items()}
    out["degree"] = degree
    out["n"] = len(out["means"])
    return out


def camera_intrinsics(scene, cam_obj):
    """Pixel size and pinhole intrinsics for the scene camera (OpenCV: y down)."""
    r = scene.render
    W = int(r.resolution_x * r.resolution_percentage / 100)
    H = int(r.resolution_y * r.resolution_percentage / 100)
    cam = cam_obj.data
    if cam.type != 'PERSP':
        raise ValueError("The Spirula engine renders perspective cameras only")
    aspect = (W * r.pixel_aspect_x) / (H * r.pixel_aspect_y)
    fit = cam.sensor_fit
    if fit == 'AUTO':
        fit = 'HORIZONTAL' if aspect >= 1 else 'VERTICAL'
    if fit == 'HORIZONTAL':
        fx = cam.lens / cam.sensor_width * W
        fy = fx * r.pixel_aspect_x / r.pixel_aspect_y
    else:
        sensor = cam.sensor_height if cam.sensor_fit == 'VERTICAL' else cam.sensor_width
        fy = cam.lens / sensor * H
        fx = fy * r.pixel_aspect_y / r.pixel_aspect_x
    big = max(W, H)
    cx = W / 2 - cam.shift_x * big
    cy = H / 2 + cam.shift_y * big
    return W, H, fx, fy, cx, cy


def view_matrix(cam_world, obj_world):
    """Object-space -> OpenCV camera space. Blender cameras look down -Z with +Y up."""
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    V = flip @ np.linalg.inv(np.array(cam_world, np.float64)) @ np.array(obj_world, np.float64)
    return V.astype(np.float32)


def srgb_to_linear(c):
    c = np.maximum(c, 0.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
