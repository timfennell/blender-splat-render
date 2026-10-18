"""Spirula render engine: Spirula Studio's 3D Gaussian splat renderer inside Blender."""

import bpy

from . import meshes
from .engine import SpirulaRenderEngine


def _slug(name):
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


_LENS_ITEMS = []
_FILM_ITEMS = []


def lens_items(self, context):
    """The game's camera bag, shelf by shelf (Blender's own camera first)."""
    if not _LENS_ITEMS:
        from . import optics
        _LENS_ITEMS.append(("NONE", "Blender Camera", "Blender's camera as set: depth of field from its "
                            "f-stop and blades, no lens character, no film"))
        shelf = None
        for l in optics.lenses():
            if l["collection"] != shelf:
                shelf = l["collection"]
                _LENS_ITEMS.append(("", shelf, ""))
            _LENS_ITEMS.append((_slug(l["name"]), l["name"], l["inspiredBy"]))
    return _LENS_ITEMS


def film_items(self, context):
    if not _FILM_ITEMS:
        from . import optics
        _FILM_ITEMS.append(("NONE", "No Film", "The render as it is (digital)"))
        for f in optics.films():
            _FILM_ITEMS.append((_slug(f["name"]), f["name"], f"ISO {f['iso']:.0f}: {f['inspiredBy']}"))
    return _FILM_ITEMS


def chosen_lens(settings):
    from . import optics
    return next((l for l in optics.lenses() if _slug(l["name"]) == settings.lens), None)


def chosen_film(settings):
    from . import optics
    return next((f for f in optics.films() if _slug(f["name"]) == settings.film), None)


def _lens_changed(self, context):
    """Mount the lens on the scene camera: its focal length (36 mm film), its f-stop range,
    its blades; depth of field on."""
    lens = chosen_lens(self)
    cam = context.scene.camera if context and context.scene else None
    if lens is None or cam is None or cam.type != 'CAMERA':
        return
    c = cam.data
    c.lens = lens["focalMM"]
    c.sensor_fit = 'HORIZONTAL'
    c.sensor_width = 36.0
    c.dof.use_dof = True
    c.dof.aperture_fstop = min(max(c.dof.aperture_fstop, lens["maxAperture"]), lens["minAperture"])
    c.dof.aperture_blades = int(lens["blades"])
    c.dof.aperture_rotation = lens["bladeRotation"]


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
    viewport_resolution: bpy.props.EnumProperty(
        name="Viewport Resolution", default='HALF',
        items=[('FULL', "Full", "Render the viewport at full resolution"),
               ('HALF', "Half", "Render the viewport at half resolution (4x faster)"),
               ('QUARTER', "Quarter", "Render the viewport at quarter resolution (16x faster)")])
    use_relight: bpy.props.BoolProperty(
        name="Relight", default=False,
        description="Light the splats with the scene's lights instead of the lighting baked into the scan")
    ambient: bpy.props.FloatProperty(
        name="Ambient", default=0.15, min=0.0, soft_max=1.0,
        description="How much of the captured lighting is kept everywhere (1 with no lights = the scan as captured)")
    exposure: bpy.props.FloatProperty(
        name="Exposure", default=0.0, soft_min=-6, soft_max=6,
        description="Brightness of the scene's lights, in stops (light units follow Cycles)")
    shine_colour: bpy.props.FloatProperty(
        name="Shine Colour", default=0.25, min=0.0, max=1.0, subtype='FACTOR',
        description="How much of the scan's captured colour the highlights take (0 = neutral highlights). "
                    "Higher keeps iridescence but can show false colour where the capture never looked")
    use_shadows: bpy.props.BoolProperty(
        name="Shadows", default=True, description="The model casts shadows onto itself")
    shadow_density: bpy.props.FloatProperty(
        name="Shadow Density", default=4.0, min=1.0, soft_max=16.0,
        description="How strongly splats block light, for shadows only. Scans often train thin walls "
                    "semi-transparent (they look solid because the walls add up), which lets light leak "
                    "through to the shadow side; raise this to stop the leak")
    viewport_shadows: bpy.props.BoolProperty(
        name="Viewport Shadows", default=False,
        description="Shadows in the viewport too (several seconds per light change on big scans; "
                    "final renders always have them)")
    shadow_resolution: bpy.props.IntProperty(
        name="Shadow Resolution", default=1024, min=128, max=4096,
        description="Size of each light's deep shadow map in pixels (finer = sharper shadows)")
    dof_samples: bpy.props.IntProperty(
        name="Depth of Field Samples", default=32, min=1, max=512,
        description="Final renders with the camera's Depth of Field on: the splats are rendered "
                    "from this many points across the lens and averaged (exact blur, fur included). "
                    "Each sample is one more splat render; relighting is done once")
    lens: bpy.props.EnumProperty(
        name="Lens", items=lens_items, update=_lens_changed,
        description="A classic 35 mm manual-focus lens from Vintage Click's bag: sets the camera's "
                    "focal length and f-stop range, and gives final renders its character "
                    "(bokeh shape, cat's eyes, swirl, glow, colour fringes, fall-off)")
    lens_tilt: bpy.props.FloatVectorProperty(
        name="Tilt", size=2, default=(0.0, 0.0), soft_min=-15, soft_max=15, subtype='XYZ',
        description="Tilt lens only: tilt of the focus plane, degrees (x across the frame, y up and down)")
    film: bpy.props.EnumProperty(
        name="Film", items=film_items,
        description="Develop final renders on a film stock from Vintage Click: its curve, colour, grain, halation")
    film_push: bpy.props.IntProperty(
        name="Push / Pull", default=0, min=-1, max=3,
        description="Rate the film this many stops faster (+, pushed: more contrast and grain) or slower (-)")
    film_exposure: bpy.props.FloatProperty(
        name="Film Exposure", default=0.0, soft_min=-4, soft_max=4,
        description="Exposure onto the film, stops over the render's own (mid grey 0.18 lands on the curve's middle)")
    film_grain: bpy.props.BoolProperty(name="Grain", default=True, description="The stock's grain")
    use_meshes: bpy.props.BoolProperty(
        name="Meshes", default=True,
        description="Render polygon objects too (path traced by Blender's Cycles inside this engine), "
                    "with shadows between them and the splats")
    use_reflections: bpy.props.BoolProperty(
        name="Reflections", default=True,
        description="Flat shiny meshes (floors, mirrors) reflect the splats in final renders, "
                    "as strongly as their material reflects (Cycles' glossy colour)")
    use_mirror_light: bpy.props.BoolProperty(
        name="Mirror Light", default=True,
        description="Light bouncing off flat mirrors and polished metal lights the splats "
                    "(each light's image in the mirror, shining through the mirror's outline)")
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
        layout.prop(s, "viewport_resolution")
        box = layout.box()
        box.label(text="Lens and Film (Vintage Click)", icon='OUTLINER_OB_CAMERA')
        box.prop(s, "lens")
        lens = chosen_lens(s)
        if lens is not None:
            from . import optics
            col = box.column(align=True)
            col.label(text=lens["inspiredBy"])
            col.label(text="f/%g - f/%g, focuses to %g m" % (lens["maxAperture"], lens["minAperture"], lens["minFocusM"]))
            mv = lens.get("movement") or {}
            if mv.get("kind") == "tilt":
                box.prop(s, "lens_tilt")
            elif mv.get("kind") == "shift":
                box.label(text="Shift: use the camera's Shift X / Y (up to %g mm = %.2f)" % (mv["most"], mv["most"] / 36))
        box.prop(s, "film")
        if chosen_film(s) is not None:
            box.prop(s, "film_push")
            box.prop(s, "film_exposure")
            box.prop(s, "film_grain")
        row = layout.row()
        cam = context.scene.camera
        row.active = cam is not None and cam.type == 'CAMERA' and cam.data.dof.use_dof
        row.prop(s, "dof_samples")
        layout.separator()
        layout.prop(s, "use_relight")
        col = layout.column()
        col.active = s.use_relight
        col.prop(s, "ambient")
        col.prop(s, "exposure")
        col.prop(s, "shine_colour", slider=True)
        col.prop(s, "use_shadows")
        sub = col.column()
        sub.active = s.use_shadows
        sub.prop(s, "shadow_density")
        sub.prop(s, "viewport_shadows")
        sub.prop(s, "shadow_resolution")
        sub.prop(s, "shadow_softness")
        layout.separator()
        layout.prop(s, "use_meshes")
        col = layout.column()
        col.active = s.use_meshes
        col.prop(s, "use_reflections")
        col.prop(s, "use_mirror_light")
        cs = getattr(context.scene, "cycles", None)
        if cs is not None:
            col.prop(cs, "device", text="Mesh Device")
            col.prop(cs, "samples", text="Mesh Samples")
            col.prop(cs, "use_denoising", text="Denoise Meshes")


class SPIRULA_PT_object(bpy.types.Panel):
    bl_label = "Spirula"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"
    COMPAT_ENGINES = {'SPIRULA'}

    @classmethod
    def poll(cls, context):
        ob = context.object
        return context.engine in cls.COMPAT_ENGINES and ob is not None and ob.type in meshes.GEOMETRY

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(context.object, "spirula_on_top")


# Panels other engines share that make sense here too.
_SHARED = ("RENDER_PT_format", "RENDER_PT_frame_range", "RENDER_PT_output", "RENDER_PT_encoding",
           "RENDER_PT_color_management", "RENDER_PT_stamp", "DATA_PT_lens", "DATA_PT_camera",
           "VIEWLAYER_PT_layer_passes", "DATA_PT_light", "DATA_PT_spot", "DATA_PT_context_light",
           # meshes are rendered by Cycles: its material, world and object panels apply
           "CYCLES_MATERIAL_PT_preview", "CYCLES_MATERIAL_PT_surface", "CYCLES_MATERIAL_PT_settings",
           "CYCLES_MATERIAL_PT_settings_surface", "MATERIAL_PT_context_material", "MATERIAL_PT_viewport",
           "CYCLES_OBJECT_PT_visibility", "CYCLES_OBJECT_PT_visibility_ray_visibility",
           "CYCLES_LIGHT_PT_light", "CYCLES_LIGHT_PT_spot", "CYCLES_LIGHT_PT_beam_shape",
           "CYCLES_WORLD_PT_surface", "CYCLES_RENDER_PT_light_paths", "CYCLES_RENDER_PT_light_paths_max_bounces",
           "CYCLES_RENDER_PT_film")


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
    bpy.types.Object.spirula_on_top = bpy.props.BoolProperty(
        name="Always on Top", default=False,
        description="Draw this object over the splats even where they're in front of it "
                    "(labels, rulers, highlight frames)")
    bpy.utils.register_class(SPIRULA_PT_object)
    _panels(True)
    meshes.register()


def unregister():
    meshes.unregister()
    _panels(False)
    bpy.utils.unregister_class(SPIRULA_PT_object)
    del bpy.types.Object.spirula_on_top
    bpy.utils.unregister_class(SPIRULA_PT_render)
    bpy.utils.unregister_class(SpirulaRenderEngine)
    del bpy.types.Scene.spirula
    bpy.utils.unregister_class(SpirulaSettings)
