"""Spirula render engine: Spirula Studio's 3D Gaussian splat renderer inside Blender."""

import bpy

from .engine import SpirulaRenderEngine


class SpirulaSettings(bpy.types.PropertyGroup):
    primitive: bpy.props.EnumProperty(
        name="Splat Model", default='3dgs',
        items=[('3dgs', "3DGS", "Classic Gaussian splatting (what most trainers use, Spirula's default)"),
               ('mip', "Mip-Splatting", "Anti-aliased 3D filter; for models trained as mip"),
               ('3dgut', "3DGUT", "Unscented-transform splatting; for models trained as 3dgut")])
    colour_input: bpy.props.EnumProperty(
        name="Model Colours", default='SRGB',
        items=[('SRGB', "sRGB", "Splats were trained on ordinary (sRGB) photos: convert to Blender's "
                                "linear space after blending, as the trainer saw them"),
               ('LINEAR', "Linear", "Splats were trained on linear images (e.g. RAW)")])
    use_relight: bpy.props.BoolProperty(
        name="Relight", default=False,
        description="Light the splats with the scene's lights instead of the lighting baked into the scan")
    ambient: bpy.props.FloatProperty(
        name="Ambient", default=0.15, min=0.0, soft_max=1.0,
        description="How much of the captured lighting is kept everywhere (1 with no lights = the scan as captured)")
    exposure: bpy.props.FloatProperty(
        name="Exposure", default=0.0, soft_min=-6, soft_max=6,
        description="Brightness of the scene's lights, in stops (light units follow Cycles)")
    use_shadows: bpy.props.BoolProperty(
        name="Shadows", default=True, description="The model casts shadows onto itself")
    shadow_resolution: bpy.props.IntProperty(
        name="Shadow Resolution", default=1024, min=128, max=4096,
        description="Size of each light's deep shadow map in pixels (finer = sharper shadows)")
    shadow_softness: bpy.props.IntProperty(
        name="Softness", default=1, min=0, max=6,
        description="Shadow edge filter radius in shadow-map pixels")


class SPIRULA_PT_render(bpy.types.Panel):
    bl_label = "Spirula"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    COMPAT_ENGINES = {'SPIRULA'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        s = context.scene.spirula
        layout.prop(s, "primitive")
        layout.prop(s, "colour_input")
        vs = context.scene.view_settings
        if s.colour_input == 'SRGB' and vs.view_transform != 'Standard':
            box = layout.box()
            box.label(text=f"View Transform is {vs.view_transform}", icon='INFO')
            box.label(text="Use Standard to match the trained colours exactly")
            box.prop(vs, "view_transform")
        layout.prop(context.scene.render, "film_transparent")
        layout.separator()
        layout.prop(s, "use_relight")
        col = layout.column()
        col.active = s.use_relight
        col.prop(s, "ambient")
        col.prop(s, "exposure")
        col.prop(s, "use_shadows")
        sub = col.column()
        sub.active = s.use_shadows
        sub.prop(s, "shadow_resolution")
        sub.prop(s, "shadow_softness")


# Panels other engines share that make sense here too.
_SHARED = ("RENDER_PT_format", "RENDER_PT_frame_range", "RENDER_PT_output", "RENDER_PT_encoding",
           "RENDER_PT_color_management", "RENDER_PT_stamp", "DATA_PT_lens", "DATA_PT_camera",
           "VIEWLAYER_PT_layer_passes", "DATA_PT_light", "DATA_PT_spot", "DATA_PT_context_light")


def _panels(add):
    for name in _SHARED:
        p = getattr(bpy.types, name, None)
        if p is not None and hasattr(p, "COMPAT_ENGINES"):
            (p.COMPAT_ENGINES.add if add else p.COMPAT_ENGINES.discard)('SPIRULA')


def register():
    bpy.utils.register_class(SpirulaSettings)
    bpy.types.Scene.spirula = bpy.props.PointerProperty(type=SpirulaSettings)
    bpy.utils.register_class(SpirulaRenderEngine)
    bpy.utils.register_class(SPIRULA_PT_render)
    _panels(True)


def unregister():
    _panels(False)
    bpy.utils.unregister_class(SPIRULA_PT_render)
    bpy.utils.unregister_class(SpirulaRenderEngine)
    del bpy.types.Scene.spirula
    bpy.utils.unregister_class(SpirulaSettings)
