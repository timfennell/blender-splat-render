"""ctypes binding to libspirula_render (Spirula Studio's 3DGS renderer as a library).

Loaded once per Blender session; the renderer keeps its GPU state and the uploaded
splats between calls, so animation frames only pay for the render itself.
"""

import ctypes
import os

import numpy as np

_LIB = None
_FP = ctypes.POINTER(ctypes.c_float)


class RenderError(RuntimeError):
    pass


LIB_NAMES = {"darwin": "libspirula_render.dylib", "linux": "libspirula_render.so",
             "win32": "spirula_render.dll"}


def library_path():
    """The renderer library for this system, shipped in lib/ (one per platform build)."""
    import sys
    here = os.path.join(os.path.dirname(__file__), "lib")
    name = LIB_NAMES.get(sys.platform)
    if name is None:
        raise RenderError(f"Spirula Render has no renderer library for {sys.platform}")
    path = os.path.join(here, name)
    if not os.path.exists(path):
        raise RenderError(f"The renderer library is missing ({path}). Install the release zip "
                          "made for this system (macOS, Windows or Linux).")
    if sys.platform == "win32":
        os.add_dll_directory(here)            # anything shipped beside it
    return path


def lib():
    global _LIB
    if _LIB is None:
        L = ctypes.CDLL(library_path())
        L.sr_last_error.restype = ctypes.c_char_p
        L.sr_set_splats.argtypes = [ctypes.c_int64, ctypes.c_int] + [_FP] * 6
        L.sr_set_splats.restype = ctypes.c_int
        L.sr_render.argtypes = [_FP] + [ctypes.c_float] * 4 + [ctypes.c_int, ctypes.c_int,
                                                                 ctypes.c_char_p, _FP, _FP, _FP]
        L.sr_render.restype = ctypes.c_int
        L.sr_render_ex.argtypes = [_FP] + [ctypes.c_float] * 4 + [ctypes.c_int, ctypes.c_int,
                                                                    ctypes.c_char_p, _FP, _FP, _FP, _FP]
        L.sr_render_ex.restype = ctypes.c_int
        L.sr_set_colors.argtypes = [_FP, _FP]
        L.sr_set_colors.restype = ctypes.c_int
        L.sr_set_lens.argtypes = [_FP, ctypes.c_int]
        L.sr_set_lens.restype = ctypes.c_int
        L.sr_set_render_degree.argtypes = [ctypes.c_int]
        L.sr_set_render_degree.restype = None
        _LIB = L
    return _LIB


def _p(a):
    return a.ctypes.data_as(_FP) if a is not None else None


def _check(code, what):
    if code:
        raise RenderError(f"{what}: {lib().sr_last_error().decode(errors='replace')}")


def set_splats(means, quats, log_scales, logit_opacities, dc, sh, degree):
    """All float32 C-contiguous: means [N,3], quats [N,4] wxyz, log_scales [N,3],
    logit_opacities [N], dc [N,3], sh [N,K,3] (coefficient-major) or None."""
    n = len(means)
    arrs = [np.ascontiguousarray(a, np.float32) if a is not None else None
            for a in (means, quats, log_scales, logit_opacities, dc, sh)]
    _check(lib().sr_set_splats(n, degree, *[_p(a) for a in arrs]), "loading splats")


def render(viewmat, fx, fy, cx, cy, width, height, primitive="3dgs"):
    """viewmat: world-to-camera 4x4, OpenCV axes. Returns rgb [H,W,3] (premultiplied,
    in the model's colour space), alpha [H,W], depth [H,W]; row 0 is the top."""
    V = np.ascontiguousarray(viewmat, np.float32).ravel()
    rgb = np.zeros((height, width, 3), np.float32)
    alpha = np.zeros((height, width), np.float32)
    depth = np.zeros((height, width), np.float32)
    _check(lib().sr_render(_p(V), fx, fy, cx, cy, width, height, primitive.encode(),
                           _p(rgb), _p(alpha), _p(depth)), "rendering")
    return rgb, alpha, depth


def set_colors(dc, sh):
    """Replace colours in place: dc [N,3], sh [N,K,3] or None (same N and degree as loaded)."""
    dc = np.ascontiguousarray(dc, np.float32)
    sh = None if sh is None else np.ascontiguousarray(sh, np.float32)
    _check(lib().sr_set_colors(_p(dc), _p(sh)), "updating colours")


def render_ex(viewmat, fx, fy, cx, cy, width, height, primitive="3dgs"):
    """render() plus the per-pixel median depth. Returns rgb, alpha, depth, median."""
    V = np.ascontiguousarray(viewmat, np.float32).ravel()
    rgb = np.zeros((height, width, 3), np.float32)
    alpha = np.zeros((height, width), np.float32)
    depth = np.zeros((height, width), np.float32)
    median = np.zeros((height, width), np.float32)
    _check(lib().sr_render_ex(_p(V), fx, fy, cx, cy, width, height, primitive.encode(),
                              _p(rgb), _p(alpha), _p(depth), _p(median)), "rendering")
    return rgb, alpha, depth, median


def set_render_degree(degree):
    """Render with at most `degree` SH bands (0 = base colour only); -1 = all loaded bands."""
    lib().sr_set_render_degree(int(degree))


def set_lens(params):
    """A thin lens inside the projection for the renders that follow (float4 table, layout in
    the spirula fork's docs/notes/lens-defocus.md; optics.lens_params builds it), or None for
    a pinhole. Process-wide: put it back to None when done."""
    if params is None:
        _check(lib().sr_set_lens(None, 0), "clearing the lens")
        return
    a = np.ascontiguousarray(params, np.float32).reshape(-1, 4)
    _check(lib().sr_set_lens(_p(a), len(a)), "setting the lens")
