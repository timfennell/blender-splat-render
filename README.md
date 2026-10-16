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
view; renders at half resolution by default). Any number of splat objects (merged per frame in world
space, so they overlap and shadow each other), perspective scene cameras, macOS (Apple Silicon).
With **Relight** on, the scene's lights light the splats, and the splats shadow themselves and each
other.

## Meshes alongside the splats

![Two bee scans on a glossy floor: shadows on the floor, reflections in it, labels and a scale bar](docs/bees_on_glossy_floor.jpg)

Polygon objects (planes, text, rulers, frames, any mesh) render with the splats. In final renders
they're path traced by Blender's own Cycles, run inside this engine, so their materials, textures
and mesh-on-mesh shadows and reflections are Cycles'. The engine joins them up with the splats:

- **Depth:** meshes and splats meet per pixel, so fur in front of a plane stays in front of it.
- **Splats shadow meshes:** each light's contribution on a mesh is dimmed by the splats' own shadow
  maps (the ones the splats shadow themselves with), so fur casts soft, see-through shadows.
- **Meshes shadow splats:** mesh triangles are drawn into each light's shadow map.
- **Reflections:** flat shiny meshes (floors, mirrors) reflect the splats, as strongly as their
  material reflects and blurred by its roughness. Curved meshes don't reflect the splats.
- **Always on Top** (Object properties > Spirula): draws a mesh over the splats even where they're
  in front of it, for labels, rulers and highlight frames.

Cycles never sees the splats (they're hidden from its rays while it renders), and its lights are
converted to this engine's units, so a light gives a mesh and a splat the same brightness. Mesh
samples, device and denoising are in the Spirula render panel.

The viewport draws a fast GPU preview of the meshes (base colour or image texture, the same lights,
the splats' shadows on them, their shadows on the splats). Mesh-on-mesh shadows and reflections are
final-render only.

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
