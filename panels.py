# This module is part of the public-release split of blender_import_model_anim_release.py.
# Behavior-sensitive logic was moved mechanically; avoid algorithm changes here.

from .utils import *

def _action_frame_summary(action):
    if not action:
        return ""
    try:
        start, end = action.frame_range
    except Exception:
        return ""
    try:
        return f"{start:.0f}-{end:.0f}"
    except Exception:
        return ""

def _count_action_keys(action, data_paths):
    if not action:
        return 0
    wanted = set(data_paths)
    total = 0
    for fc in getattr(action, "fcurves", []) or []:
        if getattr(fc, "data_path", "") in wanted:
            try:
                total += len(fc.keyframe_points)
            except Exception:
                pass
    return total

def _camera_data_action(camera):
    cam_data = getattr(camera, "data", None)
    anim_data = getattr(cam_data, "animation_data", None) if cam_data else None
    return getattr(anim_data, "action", None) if anim_data else None

def _camera_custom_track_names(action):
    if not action:
        return []
    try:
        records = json.loads(str(action.get("engine_camera_custom_tracks", "[]") or "[]"))
    except Exception:
        return []
    names = []
    for record in records if isinstance(records, list) else []:
        name = str(record.get("name", "") if isinstance(record, dict) else "")
        if name and name not in names:
            names.append(name)
    return names

def _first_idprop(targets, key, default=None):
    for target in targets:
        if not target:
            continue
        try:
            value = target.get(key)
        except Exception:
            value = None
        if value is not None:
            return value
    return default

def _draw_camera_clip_summary(layout, camera, action):
    targets = (camera, action, getattr(camera, "data", None))
    box = layout.box()
    box.label(text="Camera AnimClip", icon='CAMERA_DATA')
    col = box.column(align=True)
    col.label(text=f"Camera: {camera.name}")
    if action:
        col.label(text=f"Action: {action.name}")
        frame_summary = _action_frame_summary(action)
        if frame_summary:
            col.label(text=f"Action Frames: {frame_summary}")
    else:
        col.label(text="No camera action assigned.", icon='ERROR')

    sample_count = _first_idprop(targets, "engine_clip_import_sample_count", _first_idprop(targets, "engine_sample_cnt"))
    fps = _first_idprop(targets, "engine_clip_import_fps", _first_idprop(targets, "engine_clip_full_fps"))
    duration = _first_idprop(targets, "engine_clip_duration")
    clip_hash = _first_idprop(targets, "engine_clip_name_hash")
    motion_samples = _first_idprop(targets, "engine_motion_import_sample_count")

    if sample_count is not None:
        col.label(text=f"Imported Samples: {int(sample_count)}")
    if fps is not None and float(fps) > 0.0:
        col.label(text=f"Imported FPS: {float(fps):.3f}")
    if duration is not None and float(duration) > 0.0:
        col.label(text=f"Imported Duration: {float(duration):.3f}s")
    if motion_samples is not None and int(motion_samples) > 0:
        col.label(text=f"Motion Samples: {int(motion_samples)}")
    if clip_hash is not None:
        col.label(text=f"Clip Hash: {_hex32(clip_hash)}")

def _draw_camera_lens_settings(layout, camera):
    cam_data = getattr(camera, "data", None)
    box = layout.box()
    box.label(text="Camera Settings", icon='OUTLINER_DATA_CAMERA')
    if not cam_data:
        box.label(text="Selected object has no camera data.", icon='ERROR')
        return
    col = box.column(align=True)
    col.use_property_split = True
    col.use_property_decorate = False
    if hasattr(cam_data, "lens_unit"):
        col.prop(cam_data, "lens_unit", text="Lens Mode")
    lens_unit = str(getattr(cam_data, "lens_unit", "MILLIMETERS") or "MILLIMETERS")
    if lens_unit == "FOV" and hasattr(cam_data, "angle_x"):
        col.prop(cam_data, "angle_x", text="Horizontal FOV")
        if hasattr(cam_data, "lens"):
            col.prop(cam_data, "lens", text="Focal Length")
    else:
        if hasattr(cam_data, "lens"):
            col.prop(cam_data, "lens", text="Focal Length")
        if hasattr(cam_data, "angle_x"):
            col.prop(cam_data, "angle_x", text="Horizontal FOV")
    if hasattr(cam_data, "sensor_width"):
        col.prop(cam_data, "sensor_width", text="Sensor Width")
    if hasattr(cam_data, "clip_start"):
        col.prop(cam_data, "clip_start", text="Near Clip")
    if hasattr(cam_data, "clip_end"):
        col.prop(cam_data, "clip_end", text="Far Clip")

def _draw_camera_track_status(layout, camera, action):
    data_action = _camera_data_action(camera)
    box = layout.box()
    box.label(text="Animated Camera Data", icon='GRAPH')
    col = box.column(align=True)

    transform_keys = _count_action_keys(action, {"location", "rotation_quaternion", "rotation_euler"})
    lens_keys = _count_action_keys(data_action, {"lens"})
    fov_keys = _count_action_keys(data_action, {"angle_x"})
    track_names = _camera_custom_track_names(action)

    if transform_keys:
        col.label(text=f"Transform Keys: {transform_keys}")
    if lens_keys:
        col.label(text=f"Focal Length Keys: {lens_keys}")
    if fov_keys:
        col.label(text=f"FOV Keys: {fov_keys}")
    if track_names:
        shown = ", ".join(track_names[:4])
        suffix = f" (+{len(track_names) - 4})" if len(track_names) > 4 else ""
        col.label(text=f"Engine Tracks: {shown}{suffix}")
    if not any((transform_keys, lens_keys, fov_keys, track_names)):
        col.label(text="No imported camera keys or custom tracks found.", icon='INFO')

def _draw_camera_export_panel(layout, context, camera, action):
    scene = context.scene
    box = layout.box()
    box.label(text="Camera Export", icon='EXPORT')
    col = box.column(align=True)
    col.prop(scene, "engine_export_frame_start", text="Start Frame")
    col.prop(scene, "engine_export_frame_end", text="End Frame")
    col.prop(scene, "engine_anim_fps", text="Export FPS")
    frame_start = int(scene.engine_export_frame_start)
    frame_end = max(frame_start, int(scene.engine_export_frame_end))
    frame_count = max(1, frame_end - frame_start + 1)
    fps = max(0.001, float(scene.engine_anim_fps))
    looping = bool(scene.engine_export_looping)
    if scene.engine_export_use_original_values:
        original_flags = get_original_flags(action, camera)
        if original_flags is not None:
            looping = bool((int(original_flags) & U32_MASK) & FLAG_LOOPING)
    duration = (frame_count if looping else max(0, frame_count - 1)) / fps
    if hasattr(scene, "engine_export_duration"):
        col.prop(scene, "engine_export_duration", text="Duration")
    else:
        col.label(text=f"Export Duration: {duration:.3f}s")
    col.label(text=f"Samples: {frame_count}")
    col.prop(scene, "engine_export_use_original_values", text="Use Original File Values")
    col.prop(scene, "engine_export_add_stg_header", text="Add STG Header")

    flag_col = box.column(align=True)
    flag_col.enabled = not scene.engine_export_use_original_values
    flag_col.prop(scene, "engine_export_looping", text="Looping")
    flag_col.prop(scene, "engine_export_additive", text="Additive")
    flag_col.prop(scene, "engine_export_partial", text="Partial")
    flag_col.prop(scene, "engine_export_partial_motion", text="Partial Motion")

    trigger_count = len(_action_trigger_marker_entries(action)) if action else 0
    current_flags = get_current_flags(action, camera)
    if current_flags is None:
        current_flags = get_original_flags(action, camera)
    status = box.column(align=True)
    if current_flags is not None:
        status.label(text=f"Flags: {_hex32(current_flags)}")
    if trigger_count:
        status.label(text=f"Event Triggers: {trigger_count}", icon='EVENT')

    box.operator(ExportEngineAnim.bl_idname, text="Export Camera AnimClip", icon='EXPORT')


def _json_list_idprop(target, key):
    try:
        data = json.loads(str(target.get(key, "[]") or "[]"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _u32(value):
    try:
        return int(value) & 0xFFFFFFFF
    except Exception:
        return 0


def _fmt_bytes(value):
    try:
        value = int(value)
    except Exception:
        value = 0
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.2f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value} B"


def _draw_model_foldout(layout, target, prop_name, label):
    expanded = bool(getattr(target, prop_name, False))
    row = layout.row(align=True)
    row.prop(target, prop_name, text=("- " if expanded else "+ ") + label, toggle=True)
    return expanded


def _model_mesh_names_by_subset(arm):
    names = {}
    for obj in bpy.data.objects:
        if obj.parent != arm or obj.type != 'MESH' or obj.get("engine_bounds_type", "") == "subset_aabb":
            continue
        try:
            subset_id = int(obj.get("engine_subset_index", -1))
        except Exception:
            subset_id = -1
        if subset_id >= 0:
            names[subset_id] = obj.name
    return names


def _model_group_look_indices(group, looks):
    indices = []
    for value in group.get("look_indices", []) or []:
        try:
            look_index = int(value)
        except Exception:
            continue
        if 0 <= look_index < len(looks) and look_index not in indices:
            indices.append(look_index)
    return indices


def _model_subset_ids_for_look(look, lod_index=None):
    raw_subset_ids = look.get("subset_ids", []) or []
    if lod_index is not None:
        try:
            lods = look.get("lods", []) or []
            lod = lods[max(0, min(int(lod_index), len(lods) - 1))]
            start = int(lod.get("start", 0))
            count = int(lod.get("count", 0))
            raw_subset_ids = raw_subset_ids[start:start + count] if start >= 0 and count > 0 else []
        except Exception:
            raw_subset_ids = []
    subset_ids = []
    for value in raw_subset_ids:
        try:
            subset_id = int(value)
        except Exception:
            continue
        if subset_id >= 0 and subset_id not in subset_ids:
            subset_ids.append(subset_id)
    return subset_ids


def _draw_model_subset_list(layout, arm, subset_ids, title, limit=10):
    mesh_names = _model_mesh_names_by_subset(arm)
    col = layout.column(align=True)
    col.label(text=f"{title}: {len(subset_ids)}")
    for subset_id in subset_ids[:limit]:
        col.label(text=f"{subset_id}: {mesh_names.get(subset_id, 'Subset_' + str(subset_id))}")
    if len(subset_ids) > limit:
        col.label(text=f"+{len(subset_ids) - limit} more")


def _draw_model_source_panel(layout, arm):
    box = layout.box()
    box.label(text="Model")
    col = box.column(align=True)
    col.label(text=f"Object: {arm.name}")
    source_path = str(arm.get("engine_model_source_path", "") or "")
    if source_path:
        col.label(text=f"Source: {os.path.basename(source_path)}")
    wrapper = "STG+DAT1" if bool(arm.get("engine_model_source_had_stg", False)) else "Raw DAT1"
    col.label(text=f"Source Format: {wrapper}")
    import_mode = str(arm.get("engine_model_import_mode", "") or "")
    if import_mode:
        label = "All LODs" if import_mode == "ALL_LODS" else "LOD0 Only"
        col.label(text=f"Imported: {label}")
    if bool(arm.get("engine_model_source_has_morphs", False)):
        if bool(arm.get("engine_model_shape_keys_imported", False)):
            col.label(text=f"Deformations: Morph2 ({int(arm.get('engine_model_morph_target_count', 0))} targets)")
        else:
            col.label(text="Deformations: Morph2 not imported", icon='ERROR')
    elif bool(arm.get("engine_model_source_has_ziva", False)):
        if bool(arm.get("engine_model_ziva_shape_keys_imported", False)):
            col.label(
                text=(
                    f"Deformations: Ziva baked to shape keys "
                    f"({int(arm.get('engine_model_morph_target_count', 0))} controls)"
                )
            )
        else:
            col.label(text="Deformations: compiled Ziva")
    else:
        col.label(text="Deformations: none in source")


def _draw_model_morph_panel(layout, context, arm):
    controls = _json_list_idprop(arm, "engine_model_morph_controls_json")
    source_has_deformations = bool(
        arm.get("engine_model_source_has_morphs", False)
        or arm.get("engine_model_source_has_ziva", False)
    )
    if not controls and not source_has_deformations:
        return
    if not _draw_model_foldout(layout, arm, "engine_model_show_morphs", "Shape Keys"):
        return
    setup = layout.box()
    setup.label(text="Custom Mesh Blendshapes")
    setup.operator(MODEL_OT_create_original_blendshape_names.bl_idname, icon='SHAPEKEY_DATA')
    setup.label(text="Creates the original model's names on every mesh child of this armature.")

    box = layout.box()
    header = box.row(align=True)
    header.label(text=f"Armature Shape Key Preview: {len(controls)}")
    header.operator(MODEL_OT_sync_morph_controls.bl_idname, text="", icon='FILE_REFRESH')
    if not controls:
        box.label(text="Parent the custom meshes, then create the original blendshape names.")
        return
    box.prop(arm, "engine_model_morph_search", text="", icon='VIEWZOOM')
    search = str(getattr(arm, "engine_model_morph_search", "") or "").strip().casefold()
    filtered = [item for item in controls if search in str(item.get("name", "")).casefold()]
    col = box.column(align=True)
    previews = getattr(arm, "engine_model_morph_previews", None)
    for item in filtered[:24]:
        name = str(item.get("name", "Blendshape"))
        mesh_count = int(item.get("mesh_count", 0))
        preview_index = int(item.get("preview_index", -1))
        if previews is not None and 0 <= preview_index < len(previews):
            col.prop(previews[preview_index], "value", text=f"{name} ({mesh_count})", slider=True)
    if len(filtered) > 24:
        col.label(text=f"+{len(filtered) - 24} more; use search to narrow the list")
    elif not filtered:
        col.label(text="No matching deformation names.")


def _draw_model_dat1_panel(layout, arm):
    if not _draw_model_foldout(layout, arm, "engine_model_show_dat1", "DAT1"):
        return
    box = layout.box()
    box.label(text="DAT1")
    grid = box.grid_flow(columns=2, even_columns=True, even_rows=False, align=True)
    version = _u32(arm.get("engine_model_dat1_version", 0))
    grid.label(text=f"Version: 0x{version:08X}")
    grid.label(text=f"Size: {_fmt_bytes(arm.get('engine_model_dat1_size', 0))}")
    grid.label(text=f"Blocks: {int(arm.get('engine_model_block_count', 0))}")
    grid.label(text=f"Fixups: {int(arm.get('engine_model_fixup_count', 0))}")
    grid.label(text=f"Strings: {_fmt_bytes(arm.get('engine_model_string_table_size', 0))}")
    grid.label(text=f"Geom: {_fmt_bytes(arm.get('engine_model_geom_size', 0))}")
    grid.label(text=f"Subsets: {int(arm.get('engine_model_subset_count', 0))}")
    grid.label(text=f"Materials: {int(arm.get('engine_model_material_count', 0))}")
    grid.label(text=f"Joints: {int(arm.get('engine_model_joint_count', 0))}")
    grid.label(text=f"Geom Off: {int(arm.get('engine_model_geom_offset', 0))}")


def _draw_model_look_panel(layout, context, arm):
    looks = _json_list_idprop(arm, "engine_model_looks_json")
    groups = _json_list_idprop(arm, "engine_model_look_groups_json")
    import_all_lods = bool(arm.get("engine_model_import_all_lods", False))
    active_lod = int(getattr(arm, "active_lod", 0)) if import_all_lods else 0
    try:
        active_group = int(getattr(arm, "engine_model_active_look_group", "0"))
    except Exception:
        active_group = 0
    try:
        active_look = int(getattr(arm, "engine_model_active_look", "0"))
    except Exception:
        active_look = 0

    preview_box = layout.box()
    preview_box.label(text="Preview")
    preview_col = preview_box.column(align=True)
    if import_all_lods:
        preview_col.prop(arm, "active_lod", text="LOD")
    else:
        preview_col.label(text="LOD0 import")
    preview_col.prop(arm, "engine_model_preview_all_subsets", text="All Subsets")

    if _draw_model_foldout(layout, arm, "engine_model_show_look_groups", "Look Groups"):
        box = layout.box()
        box.label(text="Look Groups")
        box.label(text="Groups select named sets of looks; meshes live in Looks.")
        group_col = box.column(align=True)
        group_col.enabled = not bool(arm.engine_model_preview_all_subsets)
        if not groups:
            group_col.label(text="No look groups found.")
        for group_index, group in enumerate(groups):
            look_indices = _model_group_look_indices(group, looks)
            selected = bool(arm.engine_model_use_look_group) and group_index == active_group
            row = group_col.row(align=True)
            text = ("* " if selected else "") + str(group.get("name", "") or f"Group {group_index}")
            op = row.operator(MODEL_OT_select_look_group.bl_idname, text=text)
            op.group_index = group_index
            row.label(text=f"{len(look_indices)} look(s)")

        row = box.row(align=True)
        row.operator(MODEL_OT_add_look_group.bl_idname, text="New Group")
        row.operator(MODEL_OT_add_selected_to_look_group.bl_idname, text="Add Active Look")
        row.operator(MODEL_OT_remove_selected_from_look_group.bl_idname, text="Remove Active Look")

        if groups and 0 <= active_group < len(groups):
            look_indices = _model_group_look_indices(groups[active_group], looks)
            look_list = box.column(align=True)
            look_list.label(text=f"Looks In Group: {len(look_indices)}")
            for look_index in look_indices[:8]:
                name = str(looks[look_index].get("name", "") or f"Look {look_index}")
                look_list.label(text=f"{look_index}: {name}")
            if len(look_indices) > 8:
                look_list.label(text=f"+{len(look_indices) - 8} more")
            subset_ids = []
            for look_index in look_indices:
                for subset_id in _model_subset_ids_for_look(looks[look_index], active_lod):
                    if subset_id not in subset_ids:
                        subset_ids.append(subset_id)
            _draw_model_subset_list(box, arm, subset_ids, "Preview Meshes")

    if _draw_model_foldout(layout, arm, "engine_model_show_looks", "Looks"):
        look_box = layout.box()
        look_box.label(text="Looks")
        look_col = look_box.column(align=True)
        look_col.enabled = not bool(arm.engine_model_preview_all_subsets)
        if not looks:
            look_col.label(text="No looks found.")
        for look_index, look in enumerate(looks):
            subset_ids = _model_subset_ids_for_look(look, active_lod)
            selected = (not bool(arm.engine_model_use_look_group)) and look_index == active_look
            row = look_col.row(align=True)
            text = ("* " if selected else "") + str(look.get("name", "") or f"Look {look_index}")
            op = row.operator(MODEL_OT_select_look.bl_idname, text=text)
            op.look_index = look_index
            row.label(text=f"{len(subset_ids)} mesh(es)")
        row = look_box.row(align=True)
        row.operator(MODEL_OT_add_look.bl_idname, text="New Look")
        row.operator(MODEL_OT_add_selected_to_look.bl_idname, text="Add Selected Meshes")
        row.operator(MODEL_OT_remove_selected_from_look.bl_idname, text="Remove Selected Meshes")

        if looks and 0 <= active_look < len(looks):
            _draw_model_subset_list(look_box, arm, _model_subset_ids_for_look(looks[active_look], active_lod), "Meshes In Look")

        try:
            visible_ids = _model_visible_subset_ids(arm, active_lod)
        except Exception:
            visible_ids = None
        total = sum(
            1 for obj in bpy.data.objects
            if obj.parent == arm and obj.type == 'MESH' and obj.get("engine_bounds_type", "") != "subset_aabb"
        )
        if visible_ids is not None:
            look_box.label(text=f"Visible: {len(visible_ids)} / {total}")


def _draw_model_selection_panel(layout, context, arm):
    obj = getattr(context, "active_object", None)
    if not obj or getattr(obj, "parent", None) != arm or getattr(obj, "type", None) != 'MESH':
        return
    if obj.get("engine_bounds_type", "") == "subset_aabb":
        return
    box = layout.box()
    box.label(text="Subset")
    col = box.column(align=True)
    subset_index = int(obj.get("engine_subset_index", -1))
    col.label(text=f"Index: {subset_index}")
    col.label(text=f"Blender vertices: {len(getattr(obj.data, 'vertices', []))}")
    col.label(text=f"Triangle corners: {sum(len(poly.vertices) for poly in getattr(obj.data, 'polygons', []))}")
    col.label(text=f"Triangles: {len(getattr(obj.data, 'polygons', []))}")
    col.label(text=f"LOD Mask: 0x{int(obj.get('engine_lod_mask', 0)) & 0xFFFF:04X}")
    mat = obj.active_material
    if mat:
        path = str(getattr(mat, "engine_material_path", "") or mat.name)
        col.label(text=f"Material: {path}")


def _draw_model_blocks_panel(layout, arm):
    blocks = _json_list_idprop(arm, "engine_model_blocks_json")
    if not blocks:
        return
    if not _draw_model_foldout(layout, arm, "engine_model_show_blocks", "Blocks"):
        return
    box = layout.box()
    box.label(text="Blocks")
    col = box.column(align=True)
    for block in blocks[:14]:
        name = str(block.get("name", "") or f"0x{_u32(block.get('hash', 0)):08X}")
        size = _fmt_bytes(block.get("size", 0))
        col.label(text=f"{name}: {size}")
    if len(blocks) > 14:
        col.label(text=f"+{len(blocks) - 14} more")


def _draw_model_export_panel(layout, context, arm):
    box = layout.box()
    box.label(text="Export")
    col = box.column(align=True)
    col.prop(context.scene, "engine_export_add_stg_header", text="Add STG Header")
    if bool(arm.get("engine_model_source_has_morphs", False)) and not bool(arm.get("engine_model_shape_keys_imported", False)):
        col.prop(arm, "engine_model_discard_unimported_morphs", text="Discard Unimported Morph2")
    col.operator(MODEL_OT_export_with_model_settings.bl_idname, text="Export Model")


class MaterialPanel(Panel):
    bl_label = "Luna Engine Material"
    bl_idname = "MATERIAL_PT_luna_engine_material"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'material'

    @classmethod
    def poll(cls, context):
        return bool(getattr(context, "material", None))

    def draw(self, context):
        layout = self.layout
        mat = context.material
        col = layout.column(align=True)
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(mat, "engine_material_path", text="Material Path")
        col.prop(mat, "engine_material_mapping_name", text="Mapping Name")


class ModelPanel(Panel):
    bl_label = "Model"
    bl_idname = "OBJECT_PT_luna_engine_model"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Luna Engine'

    @classmethod
    def poll(cls, context):
        arm = _resolve_anim_armature(context)
        return bool(arm and arm.get("engine_model_source_path"))

    def draw(self, context):
        layout = self.layout
        arm = _resolve_anim_armature(context)
        for title, drawer in (
            ("Source", lambda: _draw_model_source_panel(layout, arm)),
            ("Deformations", lambda: _draw_model_morph_panel(layout, context, arm)),
            ("DAT1", lambda: _draw_model_dat1_panel(layout, arm)),
            ("Looks", lambda: _draw_model_look_panel(layout, context, arm)),
            ("Subset", lambda: _draw_model_selection_panel(layout, context, arm)),
            ("Export", lambda: _draw_model_export_panel(layout, context, arm)),
            ("Blocks", lambda: _draw_model_blocks_panel(layout, arm)),
        ):
            try:
                drawer()
            except Exception as exc:
                box = layout.box()
                box.label(text=f"{title}: {exc}", icon='ERROR')


class CameraAnimPanel(Panel):
    bl_label = "Camera AnimClip"
    bl_idname = "OBJECT_PT_luna_camera_anim"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Luna Engine'

    @classmethod
    def poll(cls, context):
        target = _resolve_luna_settings_target(context)
        return bool(target and getattr(target, "type", None) == 'CAMERA')

    def draw(self, context):
        layout = self.layout
        camera = _resolve_luna_settings_target(context)
        action = _target_anim_action(camera) if camera else None
        _draw_camera_clip_summary(layout, camera, action)
        _draw_camera_lens_settings(layout, camera)
        _draw_camera_track_status(layout, camera, action)
        _draw_camera_export_panel(layout, context, camera, action)

class AnimPlaybackPanel(Panel):
    bl_label = "Animation Playback"
    bl_idname = "OBJECT_PT_anim_playback"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Luna Engine'

    @classmethod
    def poll(cls, context):
        target = _resolve_luna_settings_target(context)
        return not (target and getattr(target, "type", None) == 'CAMERA')

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        active_target = _resolve_luna_settings_target(context)
        active_arm = active_target if getattr(active_target, "type", None) == 'ARMATURE' else _resolve_anim_armature(context)
        action = _target_anim_action(active_target) if active_target else _resolve_anim_action(active_arm)
        col = layout.column(align=True)
        if active_target:
            col.label(text=f"Settings Target: {active_target.name}", icon='ARMATURE_DATA' if active_target.type == 'ARMATURE' else 'CAMERA_DATA')
        col.prop(scene, "engine_anim_fps", text="Anim FPS")
        col.prop(scene, "use_smooth_playback", text="Smooth Playback (60 FPS)")
        col.label(text=f"Effective FPS: {scene.render.fps / scene.render.fps_base:.1f}")

        layout.separator()
        export_col = layout.column(align=True)
        export_col.label(text="Export")
        export_col.prop(scene, "engine_export_frame_start", text="Start Frame")
        export_col.prop(scene, "engine_export_frame_end", text="End Frame")
        export_col.prop(scene, "engine_export_add_stg_header", text="Add STG Header")
        frame_start = int(scene.engine_export_frame_start)
        frame_end = max(frame_start, int(scene.engine_export_frame_end))
        frame_count = max(1, frame_end - frame_start + 1)
        export_fps = max(0.001, float(scene.engine_anim_fps))
        looping = bool(scene.engine_export_looping)
        if scene.engine_export_use_original_values and active_target:
            original_flags = get_original_flags(action, active_target)
            if original_flags is not None:
                looping = bool((int(original_flags) & U32_MASK) & FLAG_LOOPING)
        duration = (frame_count if looping else max(0, frame_count - 1)) / export_fps
        if hasattr(scene, "engine_export_duration"):
            export_col.prop(scene, "engine_export_duration", text="Duration")
        else:
            export_col.label(text=f"Duration: {duration:.3f}s")
        export_col.label(text=f"Frames: {frame_count}")
        export_col.prop(scene, "engine_export_use_original_values", text="Use Original File Values")

        if active_target:
            _draw_anim_flags_info(export_col, scene, active_target, action, show_export_toggles=False)
        else:
            export_col.label(text="Select an armature or camera to show imported clip flags.", icon='INFO')

        flag_col = export_col.column(align=True)
        flag_col.enabled = not scene.engine_export_use_original_values
        flag_col.prop(scene, "engine_export_looping", text="Looping")
        flag_col.prop(scene, "engine_export_additive", text="Additive")
        flag_col.prop(scene, "engine_export_partial", text="Partial")
        flag_col.prop(scene, "engine_export_partial_motion", text="Partial Motion")

        layout.separator()
        _draw_root_motion_config(layout, scene, active_arm)

        layout.separator()
        export_col2 = layout.column(align=True)
        export_col2.operator(ExportEngineAnim.bl_idname, text="Export Luna Engine Anim")

        layout.separator()
        if active_arm and bool(active_arm.get("engine_model_import_all_lods", False)):
            layout.prop(active_arm, "active_lod", text="Active LOD")

class AnimEventsPanel(Panel):
    bl_label = "Animation Events"
    bl_idname = "OBJECT_PT_anim_events"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Luna Engine'

    @classmethod
    def poll(cls, context):
        obj = _resolve_anim_armature(context)
        return bool(obj and _resolve_anim_action(obj))

    def draw(self, context):
        layout = self.layout
        obj = _resolve_anim_armature(context)
        action = _resolve_anim_action(obj)
        pose_markers = getattr(action, "pose_markers", None) if action else None

        if not pose_markers:
            title_row = layout.row(align=True)
            title_row.label(text=f"Action: {action.name if action else '<none>'}")
            title_row.operator(ANIM_OT_reload_event_schemas.bl_idname, text="", icon='FILE_REFRESH')
            title_row.operator("anim.add_event", text="Add Event", icon='ADD')
            title_row.operator("anim.paste_event", text="Paste Event", icon='PASTEDOWN')
            layout.label(text="No Luna Engine events loaded.", icon='INFO')
            return

        title_row = layout.row(align=True)
        title_row.label(text=f"Action: {action.name}")
        title_row.operator(ANIM_OT_reload_event_schemas.bl_idname, text="", icon='FILE_REFRESH')
        title_row.operator("anim.add_event", text="", icon='ADD')
        title_row.operator("anim.paste_event", text="", icon='PASTEDOWN')
        layout.label(text="Edit frames and payload fields; export preserves them.")
        
        box = layout.box()
        col = box.column(align=True)
        for marker in sorted(pose_markers, key=lambda m: (m.frame, m.name)):
            m_box = col.box()

            display_name = marker.name
            marker_index = ""
            prefix = ""
            t_match = re.match(r"Trigger_\[(\d+)\]", marker.name)
            if t_match:
                idx = t_match.group(1)
                marker_index = f"#{idx} "
                prefix = f"marker_{idx}_"
                ev_hash_signed = action.get(f"{prefix}ev_hash")
                if ev_hash_signed is not None:
                    ev_hash = to_unsigned_32(ev_hash_signed)
                    if ev_hash in KNOWN_EVENT_HASHES:
                        display_name = KNOWN_EVENT_HASHES[ev_hash]



            name_row = m_box.row(align=True)
            name_row.scale_y = 0.9
            if t_match:
                change_op = name_row.operator(
                    "anim.change_event_type",
                    text=f"{marker_index}{display_name}",
                    icon='EVENT_A'
                )
                change_op.marker_name = marker.name
            else:
                name_row.label(text=display_name, icon='EVENT_A')


            ctrl_row = m_box.row(align=True)
            ctrl_row.scale_y = 0.9
            jump_op = ctrl_row.operator("anim.jump_to_frame", text="Go To Frame", icon='FRAME_NEXT')
            jump_op.frame = marker.frame
            ctrl_row.prop(marker, "frame", text="Frame")
            copy_op = ctrl_row.operator("anim.copy_event", text="", icon='COPYDOWN')
            copy_op.marker_name = marker.name
            del_op = ctrl_row.operator("anim.delete_event", text="", icon='X')
            del_op.marker_name = marker.name
# this is kinda a mess, should probably be simplified
            if t_match:
                meta_col = m_box.column(align=True)
                meta_col.use_property_split = True
                meta_col.use_property_decorate = False
                name_hash = _idprop_u32(action.get(f"{prefix}name_hash", 0), 0)
                ev_hash = _idprop_u32(action.get(f"{prefix}ev_hash", 0), 0)
                meta_col.label(text=f"Trigger Name Hash: {_hex32(name_hash)}")
                meta_col.label(text=f"Event Type Hash: {_hex32(ev_hash)}")
                event_data_off = action.get(f"{prefix}event_data_off")
                if event_data_off is not None:
                    meta_col.label(text=f"Event Data Offset: {int(event_data_off)}")
                for key, label in (("actor_hash", "Actor Target Hash"), ("loc_hash", "Locator Hash"), ("flags", "Flags"), ("rad", "Radius")):
                    prop_name = f"{prefix}{key}"
                    if prop_name in action:
                        meta_col.prop(action, idprop_path(prop_name), text=label)

                field_list_str = action.get(f"{prefix}ddl_fields", "")
                if field_list_str:
                    p_col = m_box.column(align=True)
                    p_col.use_property_split = True
                    p_col.use_property_decorate = False
                    last_group = None
                    migrated_fields = []
                    for fname in field_list_str.split(","):
                        fname = migrate_action_field_path(action, prefix, fname)
                        migrated_fields.append(fname)
                        prop_name = action_ddl_prop_name(prefix, "DDL_", fname)
                        if prop_name not in action:
                            continue
                        group_label = ddl_field_group_label(fname)
                        if group_label and group_label != last_group:
                            group_row = p_col.row(align=True)
                            group_row.label(text=group_label, icon='TRIA_DOWN')
                            last_group = group_label
                        p_row = p_col.row(align=True)
                        label = ddl_field_display_label(fname)
                        normalize_action_event_field_value(action, prefix, fname)
                        p_row.prop(action, idprop_path(prop_name), text=label)
                    if migrated_fields and ",".join(migrated_fields) != field_list_str:
                        action[f"{prefix}ddl_fields"] = ",".join(migrated_fields)
