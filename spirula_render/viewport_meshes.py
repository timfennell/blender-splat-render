"""Polygon objects in the viewport's Rendered mode: a fast GPU preview.

Final renders path trace meshes with Cycles (see meshes.py). That can't keep up with the
viewport, so here each mesh is drawn with its material's base colour (or its image
texture) lit by the scene's lights in this engine's units, and merged with the splats per
pixel by depth -- the same merge final renders use. Meshes still shadow the splats in the
viewport (the relighter gets their triangles); mesh-on-mesh shadows and reflections are
final-render only.
"""

import math

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from . import meshes

MAX_LIGHTS = 8
SHADOWED_LIGHTS = 4        # lights that get the splats' shadow on meshes in the viewport
_SHADERS = {}


def _mesh_shader():
    if "mesh" in _SHADERS:
        return _SHADERS["mesh"]
    iface = gpu.types.GPUStageInterfaceInfo("spirula_mesh_iface")
    iface.smooth('VEC3', "v_pos")
    iface.smooth('VEC3', "v_nrm")
    iface.smooth('VEC2', "v_uv")
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "view_proj")
    info.push_constant('MAT4', "view")
    info.push_constant('VEC4', "base")
    info.push_constant('INT', "use_tex")
    info.push_constant('INT', "on_top")
    info.push_constant('INT', "n_lights")
    info.push_constant('FLOAT', "ambient")
    info.push_constant('FLOAT', "gain")
    # per light: xyz = position (point/spot) or direction the light travels (sun), w = type
    info.push_constant('VEC4', "l_pos", MAX_LIGHTS)
    info.push_constant('VEC4', "l_col", MAX_LIGHTS)     # rgb * strength or power, w unused
    info.push_constant('VEC4', "l_spot", MAX_LIGHTS)    # xyz = spot axis, w = cos outer
    info.push_constant('VEC4', "l_spot2", MAX_LIGHTS)   # x = cos inner
    info.push_constant('VEC2', "screen")
    info.sampler(0, 'FLOAT_2D', "tex")
    for i in range(SHADOWED_LIGHTS):
        info.sampler(1 + i, 'FLOAT_2D', f"vis{i}")
    info.vertex_in(0, 'VEC3', "pos")
    info.vertex_in(1, 'VEC3', "nrm")
    info.vertex_in(2, 'VEC2', "uv")
    info.vertex_out(iface)
    info.fragment_out(0, 'VEC4', "out_colour")
    info.fragment_out(1, 'VEC4', "out_depth")
    info.fragment_out(2, 'VEC4', "out_pos")
    info.vertex_source('''
void main() {
    v_pos = pos;
    v_nrm = nrm;
    v_uv = uv;
    gl_Position = view_proj * vec4(pos, 1.0);
}''')
    info.fragment_source('''
void main() {
    vec3 albedo = base.rgb;
    if (use_tex != 0) albedo = texture(tex, v_uv).rgb;
    vec3 n = normalize(v_nrm);
    if (!gl_FrontFacing) n = -n;
    vec3 c = albedo * ambient;
    vec2 suv = gl_FragCoord.xy / screen;
    float shade[4] = float[4](texture(vis0, suv).r, texture(vis1, suv).r,
                              texture(vis2, suv).r, texture(vis3, suv).r);
    for (int i = 0; i < n_lights; i++) {
        vec3 l; float e;
        if (l_pos[i].w < 0.5) {               /* sun: strength 1 = the capture light */
            l = -normalize(l_pos[i].xyz);
            e = 1.0;
        } else {                              /* point / spot: power / (4 pi d^2) */
            vec3 d = l_pos[i].xyz - v_pos;
            float r2 = max(dot(d, d), 1e-8);
            l = d * inversesqrt(r2);
            e = 1.0 / (4.0 * 3.14159265 * r2);
            if (l_pos[i].w > 1.5) {
                float cs = dot(-l, normalize(l_spot[i].xyz));
                e *= clamp((cs - l_spot[i].w) / max(l_spot2[i].x - l_spot[i].w, 1e-6), 0.0, 1.0);
            }
        }
        if (i < 4) e *= shade[i];
        c += albedo * l_col[i].rgb * (gain * e * max(dot(n, l), 0.0));
    }
    out_colour = vec4(c, 1.0);
    out_pos = vec4(v_pos, 1.0);
    out_depth = vec4(-(view * vec4(v_pos, 1.0)).z, float(on_top), 0.0, 1.0);
}''')
    _SHADERS["mesh"] = gpu.shader.create_from_info(info)
    return _SHADERS["mesh"]


def _composite_shader():
    if "comp" in _SHADERS:
        return _SHADERS["comp"]
    iface = gpu.types.GPUStageInterfaceInfo("spirula_comp_iface")
    iface.smooth('VEC2', "v_uv")
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('VEC4', "background")
    info.push_constant('FLOAT', "tau")
    info.sampler(0, 'FLOAT_2D', "splat_rgba")
    info.sampler(1, 'FLOAT_2D', "splat_z")
    info.sampler(2, 'FLOAT_2D', "mesh_rgba")
    info.sampler(3, 'FLOAT_2D', "mesh_z")
    info.vertex_in(0, 'VEC2', "pos")
    info.vertex_out(iface)
    info.fragment_out(0, 'VEC4', "out_colour")
    info.vertex_source('''
void main() {
    v_uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}''')
    info.fragment_source('''
void main() {
    vec4 S = texture(splat_rgba, v_uv);
    float zs = texture(splat_z, v_uv).r;
    vec4 M = texture(mesh_rgba, v_uv);
    vec2 mz = texture(mesh_z, v_uv).rg;
    float zm = M.a > 0.0 ? mz.r : 1e30;
    vec4 s_over = S + (1.0 - S.a) * M;
    vec4 m_over = M + (1.0 - M.a) * S;
    float w = 1.0 / (1.0 + exp(clamp((zs - zm) / tau, -40.0, 40.0)));
    if (M.a > 0.0 && mz.g > 0.5) w = 0.0;      /* always on top */
    vec4 c = w * s_over + (1.0 - w) * m_over;
    c.rgb += background.rgb * (1.0 - c.a);
    c.a = 1.0;
    out_colour = c;
}''')
    _SHADERS["comp"] = gpu.shader.create_from_info(info)
    return _SHADERS["comp"]


def _material_look(mat):
    """(rgba, image or None) for a material: its Principled base colour, or the image
    texture feeding it, else its viewport display colour."""
    if mat is None:
        return (0.8, 0.8, 0.8, 1.0), None
    nt = getattr(mat, "node_tree", None)
    if nt is not None:
        for node in nt.nodes:
            if node.type == 'BSDF_PRINCIPLED' or node.type == 'BSDF_DIFFUSE':
                sock = node.inputs.get("Base Color") or node.inputs.get("Color")
                if sock is None:
                    break
                if not sock.is_linked:
                    return tuple(sock.default_value), None
                src = sock.links[0].from_node
                if src.type == 'TEX_IMAGE' and src.image is not None:
                    return (1, 1, 1, 1), src.image
                if src.type == 'TEX_CHECKER':
                    a = np.array(src.inputs["Color1"].default_value)
                    b = np.array(src.inputs["Color2"].default_value)
                    return tuple((a + b) / 2), None
                break
    return tuple(mat.diffuse_color), None


class MeshLayer:
    """Batches for the scene's meshes (rebuilt when the scene changes) and the GPU passes."""

    def __init__(self):
        self.batches = []          # (batch, rgba, image)
        self.fb = None
        self.size = None

    def rebuild(self, depsgraph):
        self.batches = []
        shader = _mesh_shader()
        for ob, M in meshes.mesh_instances(depsgraph):
            try:
                me = ob.to_mesh()
            except RuntimeError:
                continue
            try:
                me.calc_loop_triangles()
                nt = len(me.loop_triangles)
                if nt == 0:
                    continue
                co = np.empty(len(me.vertices) * 3, np.float32)
                me.vertices.foreach_get("co", co)
                tv = np.empty(nt * 3, np.int32)
                me.loop_triangles.foreach_get("vertices", tv)
                tl = np.empty(nt * 3, np.int32)
                me.loop_triangles.foreach_get("loops", tl)
                tm = np.empty(nt, np.int32)
                me.loop_triangles.foreach_get("material_index", tm)
                cn = np.empty(len(me.loops) * 3, np.float32)
                me.corner_normals.foreach_get("vector", cn)
                uv = np.zeros((len(me.loops), 2), np.float32)
                if me.uv_layers.active is not None:
                    u = np.empty(len(me.loops) * 2, np.float32)
                    me.uv_layers.active.uv.foreach_get("vector", u)
                    uv = u.reshape(-1, 2)
                mats = [s.material for s in ob.material_slots] or [None]
            finally:
                ob.to_mesh_clear()
            P = co.reshape(-1, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
            Nm = np.linalg.inv(M[:3, :3]).T
            N = cn.reshape(-1, 3).astype(np.float64) @ Nm.T
            N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
            for mi in np.unique(tm):
                sel = np.nonzero(tm == mi)[0]
                verts = tv.reshape(-1, 3)[sel].ravel()
                loops = tl.reshape(-1, 3)[sel].ravel()
                mat = mats[min(int(mi), len(mats) - 1)]
                rgba, image = _material_look(mat)
                batch = batch_for_shader(shader, 'TRIS', {
                    "pos": P[verts].astype(np.float32), "nrm": N[loops].astype(np.float32),
                    "uv": uv[loops]})
                self.batches.append((batch, rgba, image, bool(getattr(ob, "spirula_on_top", False))))

    def _framebuffer(self, W, H):
        if self.size != (W, H):
            self.colour = gpu.types.GPUTexture((W, H), format='RGBA16F')
            self.zdist = gpu.types.GPUTexture((W, H), format='RG32F')
            self.pos = gpu.types.GPUTexture((W, H), format='RGBA32F')
            self.depth = gpu.types.GPUTexture((W, H), format='DEPTH_COMPONENT32F')
            self.fb = gpu.types.GPUFrameBuffer(depth_slot=self.depth,
                                               color_slots=(self.colour, self.zdist, self.pos))
            self.white = gpu.types.GPUTexture((1, 1), format='R32F',
                                              data=gpu.types.Buffer('FLOAT', 1, [1.0]))
            self.size = (W, H)
        return self.fb

    def draw(self, rv3d, W, H, lights, ambient, gain, vis=None):
        """Draw the meshes into textures: colour (premultiplied), view depth and world
        position. vis: per light (first SHADOWED_LIGHTS), a screen texture of the light left
        after the splats' shadow, or None."""
        fb = self._framebuffer(W, H)
        shader = _mesh_shader()
        Lp, Lc, Ls, Ls2 = [], [], [], []
        for L in lights[:MAX_LIGHTS]:
            col = np.asarray(L["colour"], np.float64)
            if L["type"] == 'SUN':
                Lp.append((*L["dir"], 0.0))
                Lc.append((*(col * L["strength"]), 0.0))
            else:
                Lp.append((*L["pos"], 2.0 if L["type"] == 'SPOT' else 1.0))
                Lc.append((*(col * L["power"]), 0.0))
            cos_out, cos_in = L.get("spot_cos", (-1.0, -1.0))
            Ls.append((*L["dir"], cos_out))
            Ls2.append((cos_in, 0.0, 0.0, 0.0))
        while len(Lp) < MAX_LIGHTS:
            Lp.append((0.0,) * 4); Lc.append((0.0,) * 4); Ls.append((0.0,) * 4); Ls2.append((0.0,) * 4)
        with fb.bind():
            fb.viewport_set(0, 0, W, H)
            fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            gpu.state.blend_set('NONE')
            gpu.matrix.push_projection()
            shader.bind()
            shader.uniform_float("view_proj", rv3d.perspective_matrix)
            shader.uniform_float("view", rv3d.view_matrix)
            shader.uniform_int("n_lights", min(len(lights), MAX_LIGHTS))
            shader.uniform_float("ambient", float(ambient))
            shader.uniform_float("gain", float(gain))
            shader.uniform_float("screen", (float(W), float(H)))
            for i in range(SHADOWED_LIGHTS):
                t = vis[i] if vis is not None and i < len(vis) and vis[i] is not None else self.white
                shader.uniform_sampler(f"vis{i}", t)
            for name, rows in (("l_pos", Lp), ("l_col", Lc), ("l_spot", Ls), ("l_spot2", Ls2)):
                flat = [float(x) for v in rows for x in v]
                shader.uniform_vector_float(shader.uniform_from_name(name),
                                            gpu.types.Buffer('FLOAT', len(flat), flat), 4, MAX_LIGHTS)
            for batch, rgba, image, top in self.batches:
                shader.uniform_float("base", rgba)
                shader.uniform_int("on_top", 1 if top else 0)
                tex = None
                if image is not None:
                    try:
                        tex = gpu.texture.from_image(image)
                    except Exception:
                        tex = None
                shader.uniform_int("use_tex", 1 if tex is not None else 0)
                shader.uniform_sampler("tex", tex if tex is not None else self.colour)
                batch.draw(shader)
            gpu.matrix.pop_projection()
            gpu.state.depth_test_set('NONE')
            gpu.state.depth_mask_set(False)

    def read_positions(self, W, H):
        """World position under each pixel of the last draw: [H, W, 4], w = 1 on a mesh,
        row 0 at the bottom."""
        with self.fb.bind():
            buf = self.fb.read_color(0, 0, W, H, 4, 2, 'FLOAT')
        buf.dimensions = W * H * 4
        return np.asarray(buf, np.float32).reshape(H, W, 4)

    def composite(self, splat_tex, splat_z, background, tau, W, H):
        """Splats + meshes by depth, background behind: returns an RGBA16F texture."""
        out = gpu.types.GPUTexture((W, H), format='RGBA16F')
        fb = gpu.types.GPUFrameBuffer(color_slots=(out,))
        shader = _composite_shader()
        quad = batch_for_shader(shader, 'TRI_FAN', {"pos": ((-1, -1), (1, -1), (1, 1), (-1, 1))})
        with fb.bind():
            fb.viewport_set(0, 0, W, H)
            gpu.state.blend_set('NONE')
            shader.bind()
            shader.uniform_float("background", (*background, 1.0))
            shader.uniform_float("tau", float(tau))
            shader.uniform_sampler("splat_rgba", splat_tex)
            shader.uniform_sampler("splat_z", splat_z)
            shader.uniform_sampler("mesh_rgba", self.colour)
            shader.uniform_sampler("mesh_z", self.zdist)
            quad.draw(shader)
        return out
