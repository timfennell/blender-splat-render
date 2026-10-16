# Spirula Render (Blender 5.3+)

A Blender render engine that renders 3D Gaussian splat objects with
[Spirula Studio](https://github.com/harry7557558/spirula-studio)'s own rasterizer, the renderer the
models were trained with, instead of approximating them in EEVEE or Cycles.

Pick **Spirula (3DGS)** as the render engine; F12 and Render Animation work as usual, from the scene
camera. Splats erased with the Gaussian Splat Patch add-on are left out.

Status: early. Final renders only (no viewport preview yet), one splat object per scene, perspective
cameras, macOS (Apple Silicon). Splats only: meshes and lights are not rendered.

The renderer library is built from a branch of Spirula Studio (`blender-render`, target
`spirula_render`), which adds a small C interface (`src/blender/SplatRenderC.cpp`).
