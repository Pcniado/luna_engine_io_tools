# This module is part of the public-release split of blender_import_model_anim_release.py.
# Behavior-sensitive logic was moved mechanically; avoid algorithm changes here.

from .utils import *

LUNA_SETTINGS_TARGET_KEY = "_engine_luna_active_settings_key"
LUNA_SETTINGS_TARGET_NAME = "_engine_luna_active_settings_target_name"
LUNA_SETTINGS_PROP_PREFIX = "engine_luna_setting_"
LUNA_SETTINGS_TIMER_INTERVAL = 0.25

_LUNA_SETTINGS_SYNCING = False
_LUNA_SETTINGS_TIMER_ENABLED = False
_EXPORT_DURATION_SYNCING = False

_LUNA_SCENE_SETTING_SPECS = (
    ("engine_anim_fps", float, 30.0),
    ("use_smooth_playback", bool, False),
    ("engine_export_frame_start", int, 0),
    ("engine_export_frame_end", int, 0),
    ("engine_export_duration", float, 0.0),
    ("engine_export_add_stg_header", bool, False),
    ("engine_export_use_original_values", bool, True),
    ("engine_export_looping", bool, False),
    ("engine_export_additive", bool, False),
    ("engine_export_partial", bool, False),
    ("engine_export_partial_motion", bool, False),
    ("engine_root_motion_export_mode", str, "OVERWRITE"),
)


def _setting_idprop_name(prop_name):
    return f"{LUNA_SETTINGS_PROP_PREFIX}{prop_name}"


def _coerce_setting_value(value, value_type, default):
    try:
        if value_type is bool:
            return bool(value)
        if value_type is int:
            return int(value)
        if value_type is float:
            return float(value)
        if value_type is str:
            return str(value)
    except Exception:
        return default
    return value


def _target_anim_action(target):
    anim_data = getattr(target, "animation_data", None)
    return getattr(anim_data, "action", None) if anim_data else None


def _resolve_luna_settings_target(context=None):
    context = context or getattr(bpy, "context", None)
    if context is None:
        return None
    for attr in ("object", "active_object"):
        obj = getattr(context, attr, None)
        if obj and getattr(obj, "type", None) == "CAMERA":
            return obj
    arm = _resolve_anim_armature(context)
    if arm:
        return arm
    try:
        return _resolve_anim_camera(context)
    except Exception:
        return None


def _target_settings_key(target):
    if not target:
        return ""
    action = _target_anim_action(target)
    action_name = getattr(action, "name", "") if action else ""
    return f"{getattr(target, 'type', '')}:{getattr(target, 'name', '')}:{action_name}"


def _find_luna_settings_target(key, name=""):
    matches = []
    try:
        objects = list(bpy.data.objects)
    except Exception:
        objects = []
    for obj in objects:
        try:
            if _target_settings_key(obj) == key:
                matches.append(obj)
        except Exception:
            pass
    if matches:
        for obj in matches:
            if not name or getattr(obj, "name", "") == name:
                return obj
        return matches[0]
    if name:
        try:
            return bpy.data.objects.get(name)
        except Exception:
            return None
    return None


def _target_setting_default(target, prop_name, value_type, fallback):
    action = _target_anim_action(target)
    sources = [target, action]

    def first_number(*keys):
        for source in sources:
            if not source:
                continue
            for key in keys:
                try:
                    value = source.get(key)
                except Exception:
                    value = None
                if value is not None:
                    return value
        return None

    if prop_name == "engine_anim_fps":
        value = first_number("engine_clip_import_fps", "engine_clip_full_fps", "engine_clip_resident_fps")
        return _coerce_setting_value(value, value_type, fallback) if value is not None else fallback

    if prop_name == "engine_export_frame_start":
        value = first_number("engine_motion_import_frame_start")
        return _coerce_setting_value(value, value_type, fallback) if value is not None else 0

    if prop_name == "engine_export_frame_end":
        value = first_number("engine_motion_import_frame_end")
        if value is None:
            count = first_number("engine_clip_import_sample_count", "engine_sample_cnt", "engine_clip_full_sample_count")
            if count is not None:
                value = max(0, int(count) - 1)
        return _coerce_setting_value(value, value_type, fallback) if value is not None else fallback

    if prop_name == "engine_export_duration":
        value = first_number("engine_clip_duration")
        if value is not None:
            return _coerce_setting_value(value, value_type, fallback)
        count = first_number("engine_clip_import_sample_count", "engine_sample_cnt", "engine_clip_full_sample_count")
        fps = first_number("engine_clip_import_fps", "engine_clip_full_fps", "engine_clip_resident_fps")
        if count is not None and fps is not None and float(fps) > 0.0:
            flags = get_original_flags(action, target) or 0
            span = int(count) if (int(flags) & FLAG_LOOPING) else max(0, int(count) - 1)
            return float(span) / float(fps)
        return fallback

    if prop_name == "engine_export_use_original_values":
        has_original = any(bool(source and "engine_clip_flags_original" in source) for source in sources)
        return True if has_original else fallback

    if prop_name in {"engine_export_looping", "engine_export_additive", "engine_export_partial", "engine_export_partial_motion"}:
        flags = get_original_flags(action, target)
        if flags is not None:
            flags = int(flags) & U32_MASK
            flag_map = {
                "engine_export_looping": FLAG_LOOPING,
                "engine_export_additive": FLAG_IS_ADDITIVE,
                "engine_export_partial": FLAG_IS_PARTIAL,
                "engine_export_partial_motion": FLAG_PARTIAL_MOTION,
            }
            return bool(flags & flag_map[prop_name])

    return fallback


def store_scene_luna_settings_for_target(scene, target):
    if not scene or not target or _LUNA_SETTINGS_SYNCING:
        return
    for prop_name, value_type, default in _LUNA_SCENE_SETTING_SPECS:
        if not hasattr(scene, prop_name):
            continue
        value = _coerce_setting_value(getattr(scene, prop_name), value_type, default)
        try:
            target[_setting_idprop_name(prop_name)] = value
        except Exception:
            pass


def _apply_luna_settings_to_scene(scene, target, context=None):
    global _LUNA_SETTINGS_SYNCING
    if not scene or not target:
        return
    _LUNA_SETTINGS_SYNCING = True
    try:
        for prop_name, value_type, default in _LUNA_SCENE_SETTING_SPECS:
            if not hasattr(scene, prop_name):
                continue
            key = _setting_idprop_name(prop_name)
            try:
                value = target.get(key)
            except Exception:
                value = None
            if value is None:
                value = _target_setting_default(target, prop_name, value_type, default)
            value = _coerce_setting_value(value, value_type, default)
            try:
                setattr(scene, prop_name, value)
            except Exception:
                pass

        if hasattr(scene, "engine_root_motion_empty"):
            try:
                if getattr(target, "type", None) == "ARMATURE":
                    scene.engine_root_motion_empty = _resolve_root_motion_empty(scene, target)
                elif getattr(scene, "engine_root_motion_empty", None) is not None:
                    scene.engine_root_motion_empty = None
            except Exception:
                pass
    finally:
        _LUNA_SETTINGS_SYNCING = False

    try:
        update_smoothness(scene, context or bpy.context)
    except Exception:
        pass


def mark_luna_settings_target(scene, target):
    if not scene or not target:
        return
    try:
        scene[LUNA_SETTINGS_TARGET_KEY] = _target_settings_key(target)
        scene[LUNA_SETTINGS_TARGET_NAME] = target.name
    except Exception:
        pass


def sync_luna_settings_for_context(context=None, force=False):
    context = context or getattr(bpy, "context", None)
    if context is None:
        return None
    scene = getattr(context, "scene", None)
    target = _resolve_luna_settings_target(context)
    if not scene or not target:
        return None

    key = _target_settings_key(target)
    try:
        old_key = str(scene.get(LUNA_SETTINGS_TARGET_KEY, "") or "")
    except Exception:
        old_key = ""

    if force or key != old_key:
        try:
            old_name = str(scene.get(LUNA_SETTINGS_TARGET_NAME, "") or "")
            old_target = _find_luna_settings_target(old_key, old_name)
        except Exception:
            old_target = None
        if old_target and old_target != target:
            store_scene_luna_settings_for_target(scene, old_target)
        _apply_luna_settings_to_scene(scene, target, context)
        mark_luna_settings_target(scene, target)
    return target


def _store_active_luna_settings(context=None):
    if _LUNA_SETTINGS_SYNCING:
        return
    context = context or getattr(bpy, "context", None)
    if context is None:
        return
    scene = getattr(context, "scene", None)
    target = _resolve_luna_settings_target(context)
    if scene and target:
        store_scene_luna_settings_for_target(scene, target)
        mark_luna_settings_target(scene, target)


def _luna_settings_sync_timer():
    if not _LUNA_SETTINGS_TIMER_ENABLED:
        return None
    try:
        sync_luna_settings_for_context(bpy.context, force=False)
    except Exception:
        pass
    return LUNA_SETTINGS_TIMER_INTERVAL


def update_lod_visibility(self, context):
    armature = self
    resolve_subset_index_collisions(armature)
    sanitize_model_look_metadata(armature, mark_modified=True)
    active_lod = armature.active_lod
    if bool(armature.get("engine_model_source_path")) and not bool(armature.get("engine_model_import_all_lods", False)):
        active_lod = 0
    visible_subset_ids = _model_visible_subset_ids(armature, active_lod)
    for obj in bpy.data.objects:
        if obj.parent == armature and obj.type == 'MESH':
            if obj.get("engine_bounds_type", "") == "subset_aabb":
                continue
            if visible_subset_ids is not None:
                try:
                    visible = int(obj.get("engine_subset_index", -1)) in visible_subset_ids
                except Exception:
                    visible = False
                obj.hide_viewport = not visible
                obj.hide_render = obj.hide_viewport
                continue
            mask = obj.get("engine_lod_mask", 0)
            if mask > 0:
                visible = bool(mask & (1 << active_lod))
                obj.hide_viewport = not visible
                obj.hide_render = obj.hide_viewport


def _model_json_list(armature, key):
    try:
        data = json.loads(str(armature.get(key, "[]") or "[]"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _model_look_subset_ids(look, lod_index):
    subset_ids = look.get("subset_ids", [])
    lods = look.get("lods", [])
    try:
        lod = lods[max(0, min(int(lod_index), len(lods) - 1))]
        start = int(lod.get("start", 0))
        count = int(lod.get("count", 0))
    except Exception:
        return set()
    if not isinstance(subset_ids, list) or start < 0 or count <= 0:
        return set()
    return {int(value) for value in subset_ids[start:start + count]}


def _model_visible_subset_ids(armature, lod_index):
    if bool(getattr(armature, "engine_model_preview_all_subsets", False)):
        return {
            int(obj.get("engine_subset_index", -1))
            for obj in bpy.data.objects
            if obj.parent == armature
            and obj.type == 'MESH'
            and obj.get("engine_bounds_type", "") != "subset_aabb"
            and int(obj.get("engine_subset_index", -1)) >= 0
        }

    looks = _model_json_list(armature, "engine_model_looks_json")
    if not looks:
        return None

    if bool(getattr(armature, "engine_model_use_look_group", False)):
        groups = _model_json_list(armature, "engine_model_look_groups_json")
        group_index = int(getattr(armature, "engine_model_active_look_group", 0))
        if 0 <= group_index < len(groups):
            ids = set()
            for look_index in groups[group_index].get("look_indices", []):
                try:
                    look = looks[int(look_index)]
                except Exception:
                    continue
                ids.update(_model_look_subset_ids(look, lod_index))
            return ids

    look_index = int(getattr(armature, "engine_model_active_look", 0))
    if 0 <= look_index < len(looks):
        return _model_look_subset_ids(looks[look_index], lod_index)
    return None

def _delete_engine_aabb_helpers():
    if not hasattr(bpy.data, "objects"):
        return None
    for obj in list(bpy.data.objects):
        if obj.get("engine_bounds_type", "") == "subset_aabb" or obj.name.startswith("AABB_Subset_"):
            mesh = obj.data if obj.type == 'MESH' else None
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    if hasattr(bpy.data, "materials"):
        mat = bpy.data.materials.get("Engine_AABB_Wire")
        if mat and mat.users == 0:
            bpy.data.materials.remove(mat)
    return None

def _set_scene_fps(scene, fps):
    fps = max(0.001, float(fps))
    if fps >= 1.0:
        scene.render.fps = int(round(fps))
        scene.render.fps_base = int(round(fps)) / fps
    else:
        scene.render.fps = 1
        scene.render.fps_base = 1.0 / fps

def _scene_export_looping(scene, context=None):
    looping = bool(getattr(scene, "engine_export_looping", False))
    if bool(getattr(scene, "engine_export_use_original_values", True)) and context is not None:
        target = _resolve_luna_settings_target(context)
        if target:
            flags = get_original_flags(_target_anim_action(target), target)
            if flags is not None:
                looping = bool((int(flags) & U32_MASK) & FLAG_LOOPING)
    return looping

def _scene_export_duration(scene, context=None):
    fps = max(0.001, float(getattr(scene, "engine_anim_fps", scene.get("engine_base_fps", 30.0))))
    start = int(getattr(scene, "engine_export_frame_start", 0))
    end = max(start, int(getattr(scene, "engine_export_frame_end", start)))
    frame_count = max(1, end - start + 1)
    span = frame_count if _scene_export_looping(scene, context) else max(0, frame_count - 1)
    return float(span) / fps

def _sync_export_duration_from_range(scene, context=None):
    global _EXPORT_DURATION_SYNCING
    if not hasattr(scene, "engine_export_duration") or _EXPORT_DURATION_SYNCING or _LUNA_SETTINGS_SYNCING:
        return
    _EXPORT_DURATION_SYNCING = True
    try:
        scene.engine_export_duration = _scene_export_duration(scene, context)
    finally:
        _EXPORT_DURATION_SYNCING = False

def update_anim_fps(self, context):
    scene = self
    scene["engine_base_fps"] = float(scene.engine_anim_fps)
    update_smoothness(scene, context)
    _sync_export_duration_from_range(scene, context)
    _store_active_luna_settings(context)

def update_export_frame_range(self, context):
    global _EXPORT_DURATION_SYNCING
    scene = self
    if _LUNA_SETTINGS_SYNCING:
        return
    start = int(scene.engine_export_frame_start)
    end = int(scene.engine_export_frame_end)
    if end < start:
        scene.engine_export_frame_end = start
        end = start
    if getattr(scene, "use_smooth_playback", False):
        update_smoothness(scene, context)
    else:
        scene.frame_start = start
        scene.frame_end = end
        if scene.frame_current < start or scene.frame_current > end:
            scene.frame_set(start)
    _sync_export_duration_from_range(scene, context)
    _store_active_luna_settings(context)

def update_export_duration(self, context):
    global _EXPORT_DURATION_SYNCING
    if _EXPORT_DURATION_SYNCING or _LUNA_SETTINGS_SYNCING:
        return
    scene = self
    fps = max(0.001, float(getattr(scene, "engine_anim_fps", scene.get("engine_base_fps", 30.0))))
    duration = max(0.0, float(getattr(scene, "engine_export_duration", 0.0)))
    start = int(getattr(scene, "engine_export_frame_start", scene.frame_start))
    if _scene_export_looping(scene, context):
        frame_count = max(1, int(round(duration * fps)))
        end = start + frame_count - 1
    else:
        end = start + max(0, int(round(duration * fps)))
    _EXPORT_DURATION_SYNCING = True
    try:
        scene.engine_export_frame_end = end
        if getattr(scene, "use_smooth_playback", False):
            update_smoothness(scene, context)
        else:
            scene.frame_start = start
            scene.frame_end = end
            if scene.frame_current < start or scene.frame_current > end:
                scene.frame_set(start)
    finally:
        _EXPORT_DURATION_SYNCING = False
    _store_active_luna_settings(context)

def update_luna_scene_setting(self, context):
    if _LUNA_SETTINGS_SYNCING:
        return
    _sync_export_duration_from_range(self, context)
    _store_active_luna_settings(context)


def update_model_panel_visibility(self, context):
    try:
        update_lod_visibility(self, context)
    except Exception:
        pass


def _model_look_items(self, context):
    looks = _model_json_list(self, "engine_model_looks_json")
    if not looks:
        return [("0", "Look 0", "Default model look")]
    items = []
    for index, look in enumerate(looks):
        name = str(look.get("name", "") or f"Look {index}")
        subset_count = 0
        try:
            lods = look.get("lods", [])
            subset_count = int(lods[0].get("count", 0)) if lods else 0
        except Exception:
            subset_count = 0
        items.append((str(index), name, f"Look {index}, LOD0 subsets: {subset_count}"))
    return items


def _model_look_group_items(self, context):
    groups = _model_json_list(self, "engine_model_look_groups_json")
    if not groups:
        return [("0", "Group 0", "Default model look group")]
    items = []
    for index, group in enumerate(groups):
        name = str(group.get("name", "") or f"Group {index}")
        look_count = len(group.get("look_indices", []) or [])
        items.append((str(index), name, f"Look group {index}, looks: {look_count}"))
    return items

def update_smoothness(self, context):
    scene = context.scene
    base_fps = getattr(scene, "engine_anim_fps", scene.get("engine_base_fps", 30.0))
    factor = max(1.0, 60.0 / base_fps) if scene.use_smooth_playback else 1.0
    target_fps = base_fps * factor
    _set_scene_fps(scene, target_fps)
    scene.render.frame_map_new = int(100 * factor)
    if hasattr(scene, "engine_export_frame_start"):
        start = int(scene.engine_export_frame_start)
        end = max(start, int(scene.engine_export_frame_end))
        scene.frame_start = start
        scene.frame_end = start + int(round((end - start) * factor))
    else:
        arm = context.active_object
        if arm and arm.type == 'ARMATURE' and "engine_sample_cnt" in arm:
            scene.frame_end = int((arm["engine_sample_cnt"] - 1) * factor)
    if not _LUNA_SETTINGS_SYNCING:
        _store_active_luna_settings(context)


def register_properties():
    bpy.types.Object.active_lod = IntProperty(
        name="Active LOD", min=0, max=7, default=0, update=update_lod_visibility
    )
    bpy.types.Object.engine_model_active_look = EnumProperty(
        name="Look",
        description="Preview subsets from this ModelLook",
        items=_model_look_items,
        update=update_model_panel_visibility,
    )
    bpy.types.Object.engine_model_active_look_group = EnumProperty(
        name="Look Group",
        description="Preview looks included in this ModelLookGroup",
        items=_model_look_group_items,
        update=update_model_panel_visibility,
    )
    bpy.types.Object.engine_model_use_look_group = BoolProperty(
        name="Use Look Group",
        description="Preview a look group instead of one look",
        default=False,
        update=update_model_panel_visibility,
    )
    bpy.types.Object.engine_model_preview_all_subsets = BoolProperty(
        name="All Subsets",
        description="Show every imported subset regardless of look or LOD",
        default=False,
        update=update_model_panel_visibility,
    )
    bpy.types.Object.engine_model_show_dat1 = BoolProperty(
        name="DAT1",
        description="Show imported DAT1 metadata in the Model panel",
        default=False,
    )
    bpy.types.Object.engine_model_show_look_groups = BoolProperty(
        name="Look Groups",
        description="Show ModelLookGroup presets in the Model panel",
        default=True,
    )
    bpy.types.Object.engine_model_show_looks = BoolProperty(
        name="Looks",
        description="Show ModelLook mesh membership in the Model panel",
        default=True,
    )
    bpy.types.Object.engine_model_show_blocks = BoolProperty(
        name="Blocks",
        description="Show imported DAT1 block metadata in the Model panel",
        default=False,
    )
    bpy.types.Material.engine_material_path = StringProperty(
        name="Material Path",
        description="Engine .material asset path written into exported Model Material records",
        default="",
    )
    bpy.types.Material.engine_material_mapping_name = StringProperty(
        name="Mapping Name",
        description="Original material mapping name written into exported Model Material records",
        default="",
    )
    bpy.types.Scene.use_smooth_playback = bpy.props.BoolProperty(
        name="Smooth Playback", default=False, update=update_smoothness
    )
    bpy.types.Scene.engine_anim_fps = FloatProperty(
        name="Anim FPS", min=0.001, default=30.0, precision=3, update=update_anim_fps
    )
    bpy.types.Scene.engine_export_frame_start = IntProperty(
        name="Export Start Frame", default=0, update=update_export_frame_range
    )
    bpy.types.Scene.engine_export_frame_end = IntProperty(
        name="Export End Frame", default=0, update=update_export_frame_range
    )
    bpy.types.Scene.engine_export_duration = FloatProperty(
        name="Export Duration",
        description="Export duration in seconds; editing this adjusts the export end frame",
        min=0.0,
        default=0.0,
        precision=3,
        update=update_export_duration,
    )
    bpy.types.Scene.engine_export_add_stg_header = BoolProperty(
        name="Add STG Header", default=False, update=update_luna_scene_setting
    )
    bpy.types.Scene.engine_export_use_original_values = BoolProperty(
        name="Use Original File Values",
        description="Use imported clip metadata and flags as the export base; events, motion, and custom-track flags are still reconciled to match exported blocks",
        default=True,
        update=update_luna_scene_setting
    )
    bpy.types.Scene.engine_export_looping = BoolProperty(
        name="Looping", default=False, update=update_luna_scene_setting
    )
    bpy.types.Scene.engine_export_additive = BoolProperty(
        name="Additive", default=False, update=update_luna_scene_setting
    )
    bpy.types.Scene.engine_export_partial = BoolProperty(
        name="Partial", default=False, update=update_luna_scene_setting
    )
    bpy.types.Scene.engine_export_partial_motion = BoolProperty(
        name="Partial Motion", default=False, update=update_luna_scene_setting
    )
    bpy.types.Scene.engine_root_motion_export_mode = EnumProperty(
        name="Root Motion Export",
        items=[
            ("OVERWRITE", "Overwrite Root Motion", "Rebuild root motion from the bound Empty on every export"),
            ("INPLACE", "Remove Root Motion / In-Place", "Export without root-motion displacement"),
        ],
        default="OVERWRITE",
        update=update_luna_scene_setting
    )
    bpy.types.Scene.engine_root_motion_empty = PointerProperty(
        name="Root Motion Empty",
        description="Empty object sampled into AnimClipMotionData root motion",
        type=bpy.types.Object,
        poll=_root_motion_empty_poll,
    )
    global _LUNA_SETTINGS_TIMER_ENABLED
    _LUNA_SETTINGS_TIMER_ENABLED = True
    try:
        bpy.app.timers.register(_luna_settings_sync_timer, first_interval=LUNA_SETTINGS_TIMER_INTERVAL)
    except Exception:
        pass


def unregister_properties():
    global _LUNA_SETTINGS_TIMER_ENABLED
    _LUNA_SETTINGS_TIMER_ENABLED = False
    for owner, prop_name in (
        (bpy.types.Material, "engine_material_mapping_name"),
        (bpy.types.Material, "engine_material_path"),
        (bpy.types.Scene, "engine_root_motion_empty"),
        (bpy.types.Scene, "engine_root_motion_export_mode"),
        (bpy.types.Scene, "engine_export_partial_motion"),
        (bpy.types.Scene, "engine_export_partial"),
        (bpy.types.Scene, "engine_export_additive"),
        (bpy.types.Scene, "engine_export_looping"),
        (bpy.types.Scene, "engine_export_use_original_values"),
        (bpy.types.Scene, "engine_export_add_stg_header"),
        (bpy.types.Scene, "engine_export_duration"),
        (bpy.types.Scene, "engine_export_frame_end"),
        (bpy.types.Scene, "engine_export_frame_start"),
        (bpy.types.Scene, "engine_anim_fps"),
        (bpy.types.Scene, "use_smooth_playback"),
        (bpy.types.Object, "engine_model_show_blocks"),
        (bpy.types.Object, "engine_model_show_looks"),
        (bpy.types.Object, "engine_model_show_look_groups"),
        (bpy.types.Object, "engine_model_show_dat1"),
        (bpy.types.Object, "engine_model_preview_all_subsets"),
        (bpy.types.Object, "engine_model_use_look_group"),
        (bpy.types.Object, "engine_model_active_look_group"),
        (bpy.types.Object, "engine_model_active_look"),
        (bpy.types.Object, "active_lod"),
    ):
        if hasattr(owner, prop_name):
            delattr(owner, prop_name)

