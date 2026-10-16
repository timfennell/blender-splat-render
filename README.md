# Spirula Render (Blender 5.3+)

A Blender render engine that renders 3D Gaussian splat objects with
[Spirula Studio](https://github.com/harry7557558/spirula-studio)'s own rasterizer, the renderer the
models were trained with, instead of approximating them in EEVEE or Cycles.

![Two bee scans relit by a point light orbiting them, with shadows cast within and between the models](docs/two_bees_orbit.gif)

Two copies of a 3-million-splat bee scan, end to end, relit by a point light orbiting them (72 frames, 1280x720,
about 22 s per frame with shadows on a MacBook Pro with M2 Pro). [MP4](docs/two_bees_orbit.mp4)

Pick **Spirula (3DGS)** as the render engine; F12 and Render Animation work as usual, from the scene
camera. Splats erased with the Gaussian Splat Patch add-on are left out.

Status: early. Final renders and the viewport's Rendered mode (perspective, orthographic and camera
view; renders at half resolution by default). Any number of splat objects (merged per frame in world space, so they overlap and shadow each other), perspective scene cameras,
macOS (Apple Silicon). Splats only: meshes and lights are not rendered.

The renderer library is built from a branch of Spirula Studio (`blender-render`, target
`spirula_render`), which adds a small C interface (`src/blender/SplatRenderC.cpp`).

## Building the renderer library

`spirula_render/lib/libspirula_render.dylib` (shipped in the release zip) is built from
[Spirula Studio](https://github.com/harry7557558/spirula-studio) (GPL-3.0) at the upstream commit in
`spirula_patch/UPSTREAM_BASE.txt`, plus `spirula_patch/spirula-local-changes.patch` (which adds the C
interface `src/blender/SplatRenderC.cpp` and the `spirula_render` CMake target). Build with
`ninja spirula_render` in a Vulkan (MoltenVK) build of that tree, then copy the dylib into
`spirula_render/lib/` and run `blender --command extension build --source-dir spirula_render`.

License: GPL-3.0-or-later.
