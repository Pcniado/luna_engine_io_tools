# This module owns deterministic Blender registration for the split add-on.

from .utils import bpy
from . import (
    anim_export,
    anim_import,
    binary,
    camera_anim,
    constants,
    dat1,
    events,
    flags,
    hashes,
    model_export,
    model_import,
    model_morph,
    model_ziva,
    motion,
    operators,
    panels,
    properties,
    schemas,
    ui,
    utils,
)

_LOGIC_MODULES = (
    utils,
    hashes,
    constants,
    binary,
    dat1,
    flags,
    motion,
    camera_anim,
    schemas,
    events,
    model_import,
    model_morph,
    model_ziva,
    model_export,
    anim_import,
    anim_export,
    ui,
    operators,
    panels,
    properties,
)


def _wire_module_globals():
    namespace = {}
    for module in _LOGIC_MODULES:
        namespace.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    for module in _LOGIC_MODULES:
        vars(module).update(namespace)


def _safe_menu_remove(menu, func):
    try:
        menu.remove(func)
    except Exception:
        pass


def menu_func_import_model(self, context):
    self.layout.operator(model_import.ImportEngineModel.bl_idname, text="Luna Engine Model (.model)")


def menu_func_import_anim(self, context):
    self.layout.operator(anim_import.ImportEngineAnim.bl_idname, text="Luna Engine Anim (.animclip)")


def menu_func_export_anim(self, context):
    self.layout.operator(anim_export.ExportEngineAnim.bl_idname, text="Luna Engine Anim (.animclip)")


def menu_func_export_model(self, context):
    self.layout.operator(model_export.ExportEngineModel.bl_idname, text="Luna Engine Model (.model)")


CLASSES = (
    properties.MODEL_PG_morph_preview,
    model_import.ImportEngineModel,
    model_export.MODEL_OT_select_original_model_for_export,
    model_export.ExportEngineModel,
    anim_import.ImportEngineAnim,
    anim_export.ExportEngineAnim,
    operators.MODEL_OT_export_with_model_settings,
    operators.MODEL_OT_sync_morph_controls,
    operators.MODEL_OT_create_original_blendshape_names,
    operators.MODEL_OT_reload_ziva,
    operators.MODEL_OT_prepare_custom_ziva,
    operators.MODEL_OT_transfer_active_ziva,
    operators.MODEL_OT_transfer_all_ziva,
    operators.MODEL_OT_create_empty_ziva_targets,
    operators.MODEL_OT_capture_ziva_pose,
    operators.MODEL_OT_add_look,
    operators.MODEL_OT_add_look_group,
    operators.MODEL_OT_select_look,
    operators.MODEL_OT_select_look_group,
    operators.MODEL_OT_add_selected_to_look,
    operators.MODEL_OT_remove_selected_from_look,
    operators.MODEL_OT_add_selected_to_look_group,
    operators.MODEL_OT_remove_selected_from_look_group,
    panels.MaterialPanel,
    panels.ModelPanel,
    panels.CameraAnimPanel,
    panels.AnimPlaybackPanel,
    operators.ANIM_OT_jump_to_frame,
    operators.ANIM_OT_reload_event_schemas,
    operators.ANIM_OT_copy_event,
    operators.ANIM_OT_paste_event,
    operators.ANIM_OT_add_event,
    operators.ANIM_OT_change_event_type,
    operators.ANIM_OT_delete_event,
    operators.ANIM_OT_create_root_motion_empty,
    panels.AnimEventsPanel,
)


def register():
    _wire_module_globals()
    schemas.load_ddl_schemas()
    if hasattr(bpy.types.Object, "show_engine_aabbs"):
        del bpy.types.Object.show_engine_aabbs
    bpy.app.timers.register(properties._delete_engine_aabb_helpers, first_interval=0.1)
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    properties.register_properties()
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_model)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_anim)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export_model)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export_anim)


def unregister():
    _safe_menu_remove(bpy.types.TOPBAR_MT_file_export, menu_func_export_anim)
    _safe_menu_remove(bpy.types.TOPBAR_MT_file_export, menu_func_export_model)
    _safe_menu_remove(bpy.types.TOPBAR_MT_file_import, menu_func_import_anim)
    _safe_menu_remove(bpy.types.TOPBAR_MT_file_import, menu_func_import_model)
    properties.unregister_properties()
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


_wire_module_globals()
