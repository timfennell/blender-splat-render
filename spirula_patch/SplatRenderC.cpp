// A C interface to Spirula's 3DGS renderer, for loading into another process
// (the Blender render engine add-on calls it through ctypes).
//
// Conventions are the engine's own, i.e. a trained PLY as stored:
//   means [N,3], quats [N,4] w x y z, scales [N,3] log, opacities [N] logit,
//   features_dc [N,3] SH DC, features_sh [N,K,3] higher SH, coefficient-major
//   (K = (degree+1)^2 - 1), viewmat [16] world-to-camera, row-major, OpenCV
//   axes (x right, y down, z forward).
// Output per pixel: rgb premultiplied over transparent (no background), alpha
// = 1 - transmittance, depth along the ray.
//
// Every call returns 0 on success; sr_last_error() explains a failure.

#include <engine/Engine.h>
#include <engine/EngineState.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string g_error;
int64_t g_num = 0;
int g_degree = 0;
int g_render_degree = -1;   // -1: render with all loaded SH bands

int render_degree() {
    return g_render_degree < 0 ? g_degree : std::min(g_render_degree, g_degree);
}

TorchTensorView ttv(const void* p, std::vector<int64_t> shape) {
    return std::make_tuple((uint64_t)p, (uint32_t)4, std::move(shape));
}
TorchTensorView ttv_null() {
    return std::make_tuple((uint64_t)0, 4u, std::vector<int64_t>{0});
}

template <class F>
int guarded(F&& f) {
    try {
        g_error.clear();
        f();
        if (const char* err = backend::last_error()) {
            g_error = err;
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error";
    }
    return 1;
}

}  // namespace

// Only these functions leave the library (the engine and backend inside stay private).
#if defined(_WIN32)
#define SR_API __declspec(dllexport)
#else
#define SR_API __attribute__((visibility("default")))
#endif

extern "C" {

SR_API const char* sr_last_error() { return g_error.c_str(); }

// Load (or replace) the splats. A fresh world each time: the engine's
// set_data_3dgs is a no-op on an initialised world.
SR_API int sr_set_splats(int64_t num, int sh_degree, const float* means, const float* quats,
                  const float* scales, const float* opacities, const float* features_dc,
                  const float* features_sh) {
    return guarded([&] {
        const int64_t K = (int64_t)(sh_degree + 1) * (sh_degree + 1) - 1;
        engine_reset();
        set_data_3dgs(num, ttv(means, {num, 3}), ttv(quats, {num, 4}),
                      ttv(scales, {num, 3}), ttv(opacities, {num, 1}),
                      ttv(features_dc, {num, 3}),
                      K > 0 ? ttv(features_sh, {num, K, 3}) : ttv_null());
        g_num = num;
        g_degree = sh_degree;
    });
}

// Render one pinhole view. out_rgb [H*W*3], out_alpha / out_depth [H*W] may be null.
// primitive: "3dgs" (classic EWA splatting, what most trainers use), "mip", "3dgut".
SR_API int sr_render(const float* viewmat, float fx, float fy, float cx, float cy, int width,
              int height, const char* primitive, float* out_rgb, float* out_alpha,
              float* out_depth) {
    return guarded([&] {
        if (g_num <= 0) throw std::runtime_error("no splats loaded");
        const std::vector<float> intr = {fx, fy, cx, cy};
        const std::vector<float> dist(kCameraDistortionParams, 0.0f);
        set_camera_params(width, height, "PINHOLE", "NONE", ttv(viewmat, {1, 4, 4}),
                          ttv(intr.data(), {1, 4}),
                          ttv(dist.data(), {1, kCameraDistortionParams}));
        forward_3dgs(primitive ? primitive : "3dgs", render_degree(), false, false, 0);
        backend::device_synchronize();
        std::vector<float> Ts;
        if (out_alpha) Ts.resize((size_t)width * height);
        engine_copy_render_to_host(
            out_rgb ? ttv(out_rgb, {1, height, width, 3}) : ttv_null(),
            out_depth ? ttv(out_depth, {1, height, width, 1}) : ttv_null(),
            out_alpha ? ttv(Ts.data(), {1, height, width, 1}) : ttv_null(), ttv_null(),
            ttv_null());
        if (out_alpha)
            for (size_t i = 0; i < Ts.size(); i++) out_alpha[i] = 1.0f - Ts[i];
    });
}

// Replace the splats' colours in place (same count and SH degree as loaded):
// what relighting changes every frame, without re-uploading the geometry.
SR_API int sr_set_colors(const float* features_dc, const float* features_sh) {
    return guarded([&] {
        if (g_num <= 0) throw std::runtime_error("no splats loaded");
        auto& w = engine().world;
        const int64_t K = (int64_t)(g_degree + 1) * (g_degree + 1) - 1;
        if (features_dc)
            backend::memcpy_sync(w.features_dc.data_ptr(), features_dc, g_num * 3 * sizeof(float),
                                 backend::MemcpyKind::HostToDevice);
        if (features_sh && K > 0) {
            if (!w.features_sh.data_ptr())
                throw std::runtime_error("SH is stored quantised; colour updates need fp32 SH");
            backend::memcpy_sync(w.features_sh.data_ptr(), features_sh, g_num * K * 3 * sizeof(float),
                                 backend::MemcpyKind::HostToDevice);
        }
    });
}

// sr_render plus the per-pixel median depth (where coverage reaches 50%): the
// surface depth a shadow map wants. out_median [H*W] may be null.
SR_API int sr_render_ex(const float* viewmat, float fx, float fy, float cx, float cy, int width,
                 int height, const char* primitive, float* out_rgb, float* out_alpha,
                 float* out_depth, float* out_median) {
    return guarded([&] {
        if (g_num <= 0) throw std::runtime_error("no splats loaded");
        const std::vector<float> intr = {fx, fy, cx, cy};
        const std::vector<float> dist(kCameraDistortionParams, 0.0f);
        set_camera_params(width, height, "PINHOLE", "NONE", ttv(viewmat, {1, 4, 4}),
                          ttv(intr.data(), {1, 4}),
                          ttv(dist.data(), {1, kCameraDistortionParams}));
        forward_3dgs(primitive ? primitive : "3dgs", render_degree(), false, out_median != nullptr, 0);
        backend::device_synchronize();
        std::vector<float> Ts;
        if (out_alpha) Ts.resize((size_t)width * height);
        engine_copy_render_to_host(
            out_rgb ? ttv(out_rgb, {1, height, width, 3}) : ttv_null(),
            out_depth ? ttv(out_depth, {1, height, width, 1}) : ttv_null(),
            out_alpha ? ttv(Ts.data(), {1, height, width, 1}) : ttv_null(), ttv_null(),
            out_median ? ttv(out_median, {1, height, width, 1}) : ttv_null());
        if (out_alpha)
            for (size_t i = 0; i < Ts.size(); i++) out_alpha[i] = 1.0f - Ts[i];
    });
}

// Render with at most this many SH bands (0 = base colour only, which is what a
// fast interactive relight updates); -1 restores all loaded bands.
SR_API void sr_set_render_degree(int degree) { g_render_degree = degree; }

SR_API void sr_shutdown() { engine_reset(); g_num = 0; }

}  // extern "C"
