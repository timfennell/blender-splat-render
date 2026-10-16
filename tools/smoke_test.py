"""Load the renderer library outside Blender and render a small cloud of splats.

    python tools/smoke_test.py path/to/lib/dir

Checks that the library loads, takes splats, renders something, updates colours and returns
the median depth -- on any Vulkan device, including a software one (Mesa lavapipe) on CI.
"""

import importlib.util
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    lib_dir = os.path.abspath(sys.argv[1])
    tmp = tempfile.mkdtemp()
    shutil.copytree(lib_dir, os.path.join(tmp, "lib"))
    shutil.copy(os.path.join(HERE, "..", "spirula_render", "native.py"), tmp)
    spec = importlib.util.spec_from_file_location("native", os.path.join(tmp, "native.py"))
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)

    rng = np.random.default_rng(0)
    n = 4000
    means = rng.normal(size=(n, 3)).astype(np.float32) * 0.3
    quats = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
    log_scales = np.full((n, 3), np.log(0.03), np.float32)
    logit = np.full(n, 2.0, np.float32)
    dc = rng.normal(size=(n, 3)).astype(np.float32) * 0.5
    sh = np.zeros((n, 15, 3), np.float32)
    native.set_splats(means, quats, log_scales, logit, dc, sh, 3)
    V = np.eye(4, dtype=np.float32)
    V[2, 3] = 3.0                                # camera 3 units back, looking down +z
    rgb, alpha, depth = native.render(V, 80, 80, 32, 32, 64, 64)
    assert np.isfinite(rgb).all() and alpha.max() > 0.5, ("nothing rendered", alpha.max())
    assert 2.0 < float(np.median(depth[alpha > 0.5])) < 4.0, "depth out of range"
    native.set_colors(np.zeros_like(dc), None)
    native.set_render_degree(0)
    rgb2, alpha2, _, median = native.render_ex(V, 80, 80, 32, 32, 64, 64)
    assert abs(float(alpha2.mean()) - float(alpha.mean())) < 1e-4, "colour update changed coverage"
    assert (median[alpha2 > 0.6] > 0).all(), "median depth missing"
    print(f"OK: coverage {alpha.mean():.3f}, median depth {np.median(median[alpha2 > 0.6]):.3f}")


if __name__ == "__main__":
    main()
