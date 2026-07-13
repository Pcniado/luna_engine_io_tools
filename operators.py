# This module is part of the public-release split of blender_import_model_anim_release.py.
# Behavior-sensitive logic was moved mechanically; avoid algorithm changes here.

from .utils import *
from . import schemas

class ANIM_OT_create_root_motion_empty(Operator):
    bl_idname = "anim.create_root_motion_empty"
    bl_label = "Create Root Motion Empty"
    bl_description = "Create or bind the Anim_RT Empty and move legacy armature-object root motion onto it"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _resolve_anim_armature(context)
        if not arm:
            self.report({'ERROR'}, "Select an armature or its root-motion Empty.")
            return {'CANCELLED'}

        scene = context.scene
        action = _resolve_anim_action(arm)
        frame_start = int(getattr(scene, "engine_export_frame_start", scene.frame_start))
        frame_end = max(frame_start, int(getattr(scene, "engine_export_frame_end", scene.frame_end)))
        frames = [frame for frame in range(frame_start, frame_end + 1)]

        legacy_action = _transform_action_for_object(arm)
        legacy_has_keys = any(
            len(fc.keyframe_points) > 0 for fc in object_motion_fcurves(legacy_action, arm)
        )
        legacy_samples = []
        original_frame = scene.frame_current
        try:
            if legacy_has_keys:
                for frame in frames:
                    legacy_samples.append(_sample_object_transform(arm, frame))
        finally:
            scene.frame_set(original_frame)

        empty = _ensure_root_motion_empty(scene, arm, action, create=True, parent_armature=True)
        if not empty:
            self.report({'ERROR'}, "Could not create root-motion Empty.")
            return {'CANCELLED'}

        if legacy_samples:
            root_action = _write_root_motion_samples_to_object(empty, action, frames, legacy_samples)
            _remove_transform_fcurves(legacy_action)
            arm.location = (0.0, 0.0, 0.0)
            arm.rotation_mode = 'QUATERNION'
            arm.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            signature = compute_object_motion_signature(arm, frame_start, frame_end)
            _store_motion_metadata(
                empty, root_action, _motion_original_blob(action) or b"",
                True, "", len(legacy_samples), frame_start, frame_end, signature
            )
            self.report({'INFO'}, f"Moved {len(legacy_samples)} root-motion samples to {empty.name}.")
        else:
            self.report({'INFO'}, f"Using {empty.name} as the root-motion Empty.")
        return {'FINISHED'}

class ANIM_OT_jump_to_frame(Operator):
    bl_idname = "anim.jump_to_frame"
    bl_label = "Jump to Frame"
    bl_description = "Jump to this animation event frame"
    
    frame: IntProperty()
    
    def execute(self, context):
        context.scene.frame_set(self.frame)
        return {'FINISHED'}

class ANIM_OT_reload_event_schemas(Operator):
    bl_idname = "anim.reload_event_schemas"
    bl_label = "Reload Event Schemas"
    bl_description = "Reload event and field names from JSON symbol/schema caches"

    def execute(self, context):
        schemas.load_ddl_schemas(force=True)
        self.report({'INFO'}, schemas.SCHEMA_LOAD_STATUS)
        return {'FINISHED'}


class MODEL_OT_export_with_model_settings(Operator):
    bl_idname = "model.export_with_luna_settings"
    bl_label = "Export Luna Engine Model"
    bl_description = "Export the selected model using the Model panel DAT1/STG setting"
    bl_options = {'REGISTER'}

    def execute(self, context):
        mode = "STG" if bool(getattr(context.scene, "engine_export_add_stg_header", True)) else "RAW"
        return bpy.ops.export_scene.engine_model('INVOKE_DEFAULT', stg_mode=mode)


class MODEL_OT_sync_morph_controls(Operator):
    bl_idname = "model.sync_morph_controls"
    bl_label = "Refresh Deformation Controls"
    bl_description = "Connect same-named registered shape keys on every model subset to shared root-armature sliders"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            self.report({'ERROR'}, "Select the model armature or one of its mesh children.")
            return {'CANCELLED'}
        try:
            controls = sync_armature_morph_controls(arm)
        except Exception as exc:
            self.report({'ERROR'}, f"Could not refresh deformation controls: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Connected {len(controls)} model-level deformation control(s).")
        return {'FINISHED'}


def _original_model_blendshape_channels(arm):
    source_path = str(arm.get("engine_model_source_path", "") or "")
    if not source_path or not os.path.isfile(source_path):
        raise ValueError("the original imported .model file is unavailable")

    channels = []
    data, blocks, _string_base = get_dat1_data(source_path)
    if not data:
        raise ValueError("the original .model does not contain a valid DAT1")
    morph = decode_model_morph2(data, blocks)
    if morph is not None:
        for target in morph.get("targets", []):
            channels.append({
                "index": int(target.get("index", -1)),
                "name": str(target.get("name", "") or ""),
                "hash": int(target.get("hash", 0)) & U32_MASK,
                "kind": "MORPH2",
                "driver_status": "AUTOMATIC_MORPH",
            })

    ziva_model = load_ziva_model(source_path)
    if ziva_model is not None:
        channels.extend(dict(channel) for channel in ziva_model.sliders)

    unique = []
    names = {}
    hashes = {}
    for channel in channels:
        name = str(channel.get("name", "") or "").strip()
        name_hash = int(channel.get("hash", 0)) & U32_MASK
        if not name:
            continue
        previous_hash = names.get(name)
        if previous_hash is not None:
            if previous_hash != name_hash:
                raise ValueError(f"blendshape {name!r} has conflicting source hashes")
            continue
        previous_name = hashes.get(name_hash)
        if previous_name is not None and previous_name != name:
            raise ValueError(f"blendshape hash collision: {previous_name!r} and {name!r}")
        names[name] = name_hash
        hashes[name_hash] = name
        unique.append(channel)
    if not unique:
        raise ValueError("the original .model contains no named Morph2 or Ziva blendshapes")
    return unique


class MODEL_OT_create_original_blendshape_names(Operator):
    bl_idname = "model.create_original_blendshape_names"
    bl_label = "Create Original Blendshape Names"
    bl_description = "Read the original .model and create its named blendshapes as ordinary Blender shape keys on every mesh under this armature"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            self.report({'ERROR'}, "Select the imported model armature.")
            return {'CANCELLED'}
        meshes = [
            obj for obj in bpy.data.objects
            if obj.type == 'MESH'
            and obj.parent == arm
            and len(obj.data.vertices)
            and obj.get("engine_bounds_type", "") != "subset_aabb"
        ]
        if not meshes:
            self.report({'ERROR'}, "Parent at least one custom mesh directly to the imported armature first.")
            return {'CANCELLED'}
        try:
            channels = _original_model_blendshape_channels(arm)
            result = create_empty_ziva_targets(meshes, channels, overwrite=False)
            arm["engine_ziva_mode"] = "CUSTOM_MORPH2"
            arm["engine_model_shape_keys_imported"] = True
            arm["engine_model_morph_target_count"] = len(channels)
            controls = sync_armature_morph_controls(arm)
        except Exception as exc:
            self.report({'ERROR'}, f"Could not create original blendshapes: {exc}")
            return {'CANCELLED'}
        self.report(
            {'INFO'},
            f"Created {result['created']} ordinary shape key(s) on {len(meshes)} mesh(es); "
            f"{result['existing']} already existed; {len(controls)} armature preview slider(s).",
        )
        return {'FINISHED'}


def _ziva_model_for_armature(arm, force=False):
    source_path = str(arm.get("engine_model_source_path", "") or "")
    if not source_path or not os.path.isfile(source_path):
        raise ValueError("the imported source .model file is unavailable")
    if force:
        clear_ziva_cache()
    model = load_ziva_model(source_path, use_cache=not force)
    if model is None:
        raise ValueError("the source model does not contain compiled Ziva data")
    arm["engine_ziva_metadata_json"] = json.dumps(
        model.metadata(), separators=(",", ":"), sort_keys=True
    )
    return model


def _ziva_selected_meshes(context, arm):
    result = []
    for obj in list(getattr(context, "selected_objects", []) or []):
        if obj.type == 'MESH' and obj.parent == arm and obj.get("engine_bounds_type", "") != "subset_aabb":
            result.append(obj)
    active = getattr(context, "active_object", None)
    if active and active.type == 'MESH' and active.parent == arm and active not in result:
        result.append(active)
    return result


def _ziva_active_channel(arm, model):
    value = str(getattr(arm, "engine_ziva_active_channel", "NONE") or "NONE")
    try:
        index = int(value)
    except Exception:
        raise ValueError("this source has no named Ziva channels")
    if not 0 <= index < len(model.sliders):
        raise ValueError("the selected Ziva channel is no longer valid; reload its metadata")
    return model.sliders[index]


class MODEL_OT_reload_ziva(Operator):
    bl_idname = "model.reload_ziva"
    bl_label = "Reload Ziva Source"
    bl_description = "Re-read and validate the compiled Ziva solver from the imported source model"
    bl_options = {'REGISTER'}

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            self.report({'ERROR'}, "Select the imported model armature or one of its mesh children.")
            return {'CANCELLED'}
        try:
            model = _ziva_model_for_armature(arm, force=True)
        except Exception as exc:
            self.report({'ERROR'}, f"Could not load Ziva source: {exc}")
            return {'CANCELLED'}
        self.report(
            {'INFO'},
            f"Loaded {model.elem_count} element(s), {len(model.sliders)} named channel(s), "
            f"and {model.joint_lookup_count} joint input(s).",
        )
        return {'FINISHED'}


class MODEL_OT_prepare_custom_ziva(Operator):
    bl_idname = "model.prepare_custom_ziva"
    bl_label = "Prepare Custom Ziva Replacement"
    bl_description = "Use editable Morph2 targets for custom meshes instead of exporting the source fixed-topology Ziva solver"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            self.report({'ERROR'}, "Select the imported model armature or a parented replacement mesh.")
            return {'CANCELLED'}
        try:
            model = _ziva_model_for_armature(arm)
        except Exception as exc:
            self.report({'ERROR'}, f"Could not prepare Ziva conversion: {exc}")
            return {'CANCELLED'}
        arm["engine_ziva_mode"] = "CUSTOM_MORPH2"
        selected = _ziva_selected_meshes(context, arm)
        for obj in selected:
            element = int(getattr(obj, "engine_ziva_element_index", obj.get("engine_ziva_element_index", 0)))
            if not 0 <= element < model.elem_count:
                obj.engine_ziva_element_index = 0
        self.report(
            {'INFO'},
            f"Custom Morph2 conversion enabled. Select parented meshes and transfer {len(model.sliders)} named channel(s).",
        )
        return {'FINISHED'}


class _MODEL_OT_transfer_ziva_base:
    transfer_active_only = False

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            self.report({'ERROR'}, "Select the imported armature or a parented replacement mesh.")
            return {'CANCELLED'}
        objects = _ziva_selected_meshes(context, arm)
        if not objects:
            self.report({'ERROR'}, "Select at least one nonempty mesh parented directly to the model armature.")
            return {'CANCELLED'}
        try:
            model = _ziva_model_for_armature(arm)
            channels = [_ziva_active_channel(arm, model)] if self.transfer_active_only else list(model.sliders)
            arm["engine_ziva_mode"] = "CUSTOM_MORPH2"
            wm = context.window_manager
            wm.progress_begin(0, max(1, len(channels)))
            try:
                result = transfer_ziva_channels_to_objects(
                    model,
                    arm,
                    objects,
                    channels=channels,
                    overwrite=bool(getattr(arm, "engine_ziva_overwrite_targets", False)),
                    max_distance=float(getattr(arm, "engine_ziva_transfer_distance", 0.05)),
                    progress=lambda current, total, channel: wm.progress_update(current),
                )
            finally:
                wm.progress_end()
            controls = sync_armature_morph_controls(arm)
        except Exception as exc:
            self.report({'ERROR'}, f"Ziva transfer failed: {exc}")
            return {'CANCELLED'}
        unmapped = sum(int(item["unmapped"]) for item in result["objects"].values())
        external = sum(
            channel.get("driver_status") in {"GAME_BRIDGE_REQUIRED", "EXTERNAL"}
            for channel in channels
        )
        suffix = f"; {external} need a Ziva-track-to-Morph2 game bridge" if external else ""
        self.report(
            {'WARNING'} if unmapped else {'INFO'},
            f"Created {result['created']} target instance(s), {result['empty']} empty, "
            f"{unmapped} unmapped vertices; {len(controls)} shared control(s){suffix}.",
        )
        return {'FINISHED'}


class MODEL_OT_transfer_active_ziva(_MODEL_OT_transfer_ziva_base, Operator):
    bl_idname = "model.transfer_active_ziva"
    bl_label = "Transfer Selected Channel"
    bl_description = "Evaluate the selected compiled Ziva channel and transfer its displacement to selected replacement meshes"
    bl_options = {'REGISTER', 'UNDO'}
    transfer_active_only = True


class MODEL_OT_transfer_all_ziva(_MODEL_OT_transfer_ziva_base, Operator):
    bl_idname = "model.transfer_all_ziva"
    bl_label = "Transfer All Named Channels"
    bl_description = "Evaluate every named compiled Ziva channel and transfer their displacements to selected replacement meshes"
    bl_options = {'REGISTER', 'UNDO'}


class MODEL_OT_create_empty_ziva_targets(Operator):
    bl_idname = "model.create_empty_ziva_targets"
    bl_label = "Create Empty Named Targets"
    bl_description = "Create registered same-named shape keys for manual sculpting while preserving the game's channel hashes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            self.report({'ERROR'}, "Select the imported armature or a parented replacement mesh.")
            return {'CANCELLED'}
        objects = _ziva_selected_meshes(context, arm)
        if not objects:
            self.report({'ERROR'}, "Select at least one mesh parented directly to the model armature.")
            return {'CANCELLED'}
        try:
            model = _ziva_model_for_armature(arm)
            if not model.sliders:
                raise ValueError("this Ziva rig has no named slider channels; use pose capture for joint-only deformation")
            result = create_empty_ziva_targets(
                objects,
                model.sliders,
                overwrite=bool(getattr(arm, "engine_ziva_overwrite_targets", False)),
            )
            arm["engine_ziva_mode"] = "CUSTOM_MORPH2"
            controls = sync_armature_morph_controls(arm)
        except Exception as exc:
            self.report({'ERROR'}, f"Could not create targets: {exc}")
            return {'CANCELLED'}
        self.report(
            {'INFO'},
            f"Created {result['created']} empty target(s); {result['existing']} already existed; "
            f"{len(controls)} shared control(s).",
        )
        return {'FINISHED'}


class MODEL_OT_capture_ziva_pose(Operator):
    bl_idname = "model.capture_ziva_pose"
    bl_label = "Capture Current Ziva Pose"
    bl_description = "Evaluate the compiled joint-driven Ziva solver at the current armature pose and transfer it to a registered Morph2 target"
    bl_options = {'REGISTER', 'UNDO'}

    target_name: StringProperty(
        name="Target Name",
        description="Morph2 target name and game-visible CRC32 identity; a new name requires a matching game-side driver",
        default="",
    )

    def invoke(self, context, event):
        if not self.target_name:
            self.target_name = f"ziva_pose_{int(context.scene.frame_current)}"
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        col = self.layout.column(align=True)
        col.prop(self, "target_name")
        col.label(text="Captured targets need a matching game-side driver or an existing known hash.", icon='INFO')

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            self.report({'ERROR'}, "Select the imported armature or a parented replacement mesh.")
            return {'CANCELLED'}
        objects = _ziva_selected_meshes(context, arm)
        if not objects:
            self.report({'ERROR'}, "Select at least one mesh parented directly to the model armature.")
            return {'CANCELLED'}
        try:
            model = _ziva_model_for_armature(arm)
            names = [""] * model.model_joint_count
            for bone in arm.data.bones:
                joint_index = int(bone.get("engine_joint_index", -1))
                if 0 <= joint_index < len(names):
                    names[joint_index] = bone.name
            local_matrices = _compute_frame_engine_locals(
                arm,
                names,
                model.joint_parents,
                int(context.scene.frame_current),
            )
            result = transfer_ziva_pose_to_objects(
                model,
                arm,
                objects,
                local_matrices,
                self.target_name,
                overwrite=bool(getattr(arm, "engine_ziva_overwrite_targets", False)),
                max_distance=float(getattr(arm, "engine_ziva_transfer_distance", 0.05)),
            )
            arm["engine_ziva_mode"] = "CUSTOM_MORPH2"
            controls = sync_armature_morph_controls(arm)
        except Exception as exc:
            self.report({'ERROR'}, f"Could not capture Ziva pose: {exc}")
            return {'CANCELLED'}
        unmapped = sum(int(item["unmapped"]) for item in result["objects"].values())
        self.report(
            {'WARNING'},
            f"Captured {result['created']} target instance(s), {result['empty']} empty, "
            f"{unmapped} unmapped vertices; {len(controls)} shared control(s). A game-side driver is required.",
        )
        return {'FINISHED'}


def _model_armature_from_context(context):
    return _resolve_anim_armature(context)


def _model_json_list_for_edit(arm, key):
    try:
        data = json.loads(str(arm.get(key, "[]") or "[]"))
    except Exception:
        data = []
    return data if isinstance(data, list) else []


def _write_model_json_list(arm, key, data):
    arm[key] = json.dumps(data, separators=(",", ":"))


def _selected_model_subset_ids(context, arm, assign_missing=True):
    if assign_missing:
        resolve_subset_index_collisions(arm)
    used_ids = set()
    for obj in bpy.data.objects:
        if obj.parent != arm or obj.type != 'MESH' or obj.get("engine_bounds_type", "") == "subset_aabb":
            continue
        try:
            subset_id = int(obj.get("engine_subset_index", -1))
        except Exception:
            subset_id = -1
        if subset_id >= 0:
            used_ids.add(subset_id)
    next_id = (max(used_ids) + 1) if used_ids else 0
    ids = []
    selected_objs = []

    def add_selected_obj(obj):
        if obj and obj not in selected_objs:
            selected_objs.append(obj)

    for obj in getattr(context, "selected_objects", []) or []:
        add_selected_obj(obj)
    for attr in ("object", "active_object"):
        obj = getattr(context, attr, None)
        add_selected_obj(obj)
    if hasattr(context, "selected_ids"):
        for id_data in context.selected_ids:
            if isinstance(id_data, bpy.types.Object):
                add_selected_obj(id_data)
    view_layer = getattr(context, "view_layer", None)
    for obj in bpy.data.objects:
        try:
            is_selected = obj.select_get(view_layer=view_layer) if view_layer else obj.select_get()
        except Exception:
            is_selected = False
        if is_selected:
            add_selected_obj(obj)

    for obj in selected_objs:
        if obj.parent != arm or obj.type != 'MESH' or obj.get("engine_bounds_type", "") == "subset_aabb":
            continue
        try:
            subset_id = int(obj.get("engine_subset_index", -1))
        except Exception:
            subset_id = -1
        if subset_id < 0 and assign_missing:
            while next_id in used_ids:
                next_id += 1
            obj["engine_subset_index"] = next_id
            obj["engine_lod_mask"] = int(obj.get("engine_lod_mask", 1) or 1)
            subset_id = next_id
            used_ids.add(subset_id)
            next_id += 1
        if subset_id >= 0 and subset_id not in ids:
            ids.append(subset_id)
    return ids


def _model_all_subset_ids(arm):
    return model_mesh_subset_ids(arm)


def _source_model_subset_count(arm):
    try:
        return int(arm.get("engine_model_subset_count", 0) or 0)
    except Exception:
        return 0


def _custom_model_subset_ids(arm, subset_ids):
    source_count = _source_model_subset_count(arm)
    return {int(value) for value in subset_ids if int(value) >= source_count}


def _normalize_model_look(arm, look, subset_ids):
    valid_ids = set(_model_all_subset_ids(arm))
    subset_ids = [int(value) for value in subset_ids if int(value) in valid_ids]
    look["subset_ids"] = subset_ids
    look["lods"] = [{"start": 0, "count": len(subset_ids)} for _ in range(8)]


def _set_model_look_name(look, name, fallback_index=0):
    clean = str(name or "").strip() or f"Look {fallback_index}"
    look["name"] = clean
    look["name_hash"] = int(string_crc32(clean))


def _set_model_group_name(group, name, fallback_index=0):
    clean = str(name or "").strip() or f"Look Group {fallback_index}"
    group["name"] = clean
    group["name_hash"] = int(string_crc32(clean))


def _active_model_look_index(arm):
    try:
        return int(getattr(arm, "engine_model_active_look", "0"))
    except Exception:
        return 0


def _active_model_group_index(arm):
    try:
        return int(getattr(arm, "engine_model_active_look_group", "0"))
    except Exception:
        return 0


class MODEL_OT_add_look(Operator):
    bl_idname = "model.add_luna_look"
    bl_label = "Add Model Look"
    bl_description = "Create a new ModelLook from selected meshes, or all meshes when none are selected"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty(name="Name", default="")

    def invoke(self, context, event):
        arm = _model_armature_from_context(context)
        looks = _model_json_list_for_edit(arm, "engine_model_looks_json") if arm else []
        if not self.name:
            self.name = f"Look {len(looks)}"
        return context.window_manager.invoke_props_dialog(self, width=360)

    def draw(self, context):
        self.layout.prop(self, "name")

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            self.report({'ERROR'}, "Select an imported model armature or mesh.")
            return {'CANCELLED'}
        sanitize_model_look_metadata(arm, mark_modified=True)
        looks = _model_json_list_for_edit(arm, "engine_model_looks_json")
        subset_ids = _selected_model_subset_ids(context, arm, assign_missing=True) or _model_all_subset_ids(arm)
        look_index = len(looks)
        look = {
            "index": look_index,
            "subset_ids": [],
            "lods": [],
        }
        _set_model_look_name(look, self.name, look_index)
        _normalize_model_look(arm, look, subset_ids)
        looks.append(look)
        _write_model_json_list(arm, "engine_model_looks_json", looks)
        arm["engine_model_looks_modified"] = True
        arm.engine_model_active_look = str(look_index)
        arm.engine_model_use_look_group = False
        update_lod_visibility(arm, context)
        self.report({'INFO'}, f"Added Look {look_index} with {len(subset_ids)} subset(s).")
        return {'FINISHED'}


class MODEL_OT_add_look_group(Operator):
    bl_idname = "model.add_luna_look_group"
    bl_label = "Add Model Look Group"
    bl_description = "Create a new ModelLookGroup containing the active look"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty(name="Name", default="")

    def invoke(self, context, event):
        arm = _model_armature_from_context(context)
        groups = _model_json_list_for_edit(arm, "engine_model_look_groups_json") if arm else []
        if not self.name:
            self.name = f"Look Group {len(groups)}"
        return context.window_manager.invoke_props_dialog(self, width=360)

    def draw(self, context):
        self.layout.prop(self, "name")

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            self.report({'ERROR'}, "Select an imported model armature or mesh.")
            return {'CANCELLED'}
        sanitize_model_look_metadata(arm, mark_modified=True)
        looks = _model_json_list_for_edit(arm, "engine_model_looks_json")
        if not looks:
            self.report({'ERROR'}, "No looks exist on this model.")
            return {'CANCELLED'}
        look_index = max(0, min(_active_model_look_index(arm), len(looks) - 1))
        groups = _model_json_list_for_edit(arm, "engine_model_look_groups_json")
        group_index = len(groups)
        group = {
            "index": group_index,
            "look_indices": [look_index],
        }
        _set_model_group_name(group, self.name, group_index)
        groups.append(group)
        _write_model_json_list(arm, "engine_model_look_groups_json", groups)
        arm["engine_model_looks_modified"] = True
        arm.engine_model_active_look_group = str(group_index)
        arm.engine_model_use_look_group = True
        update_lod_visibility(arm, context)
        self.report({'INFO'}, f"Added Look Group {group_index}.")
        return {'FINISHED'}


class MODEL_OT_select_look(Operator):
    bl_idname = "model.select_luna_look"
    bl_label = "Select Model Look"
    bl_description = "Select this ModelLook for preview and editing"
    bl_options = {'REGISTER', 'UNDO'}

    look_index: IntProperty(default=0)

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            return {'CANCELLED'}
        sanitize_model_look_metadata(arm, mark_modified=True)
        arm.engine_model_active_look = str(max(0, int(self.look_index)))
        arm.engine_model_use_look_group = False
        update_lod_visibility(arm, context)
        return {'FINISHED'}


class MODEL_OT_select_look_group(Operator):
    bl_idname = "model.select_luna_look_group"
    bl_label = "Select Model Look Group"
    bl_description = "Select this ModelLookGroup for preview"
    bl_options = {'REGISTER', 'UNDO'}

    group_index: IntProperty(default=0)

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            return {'CANCELLED'}
        sanitize_model_look_metadata(arm, mark_modified=True)
        arm.engine_model_active_look_group = str(max(0, int(self.group_index)))
        arm.engine_model_use_look_group = True
        update_lod_visibility(arm, context)
        return {'FINISHED'}


class MODEL_OT_add_selected_to_look(Operator):
    bl_idname = "model.add_selected_to_luna_look"
    bl_label = "Add Selected Meshes"
    bl_description = "Add selected mesh subsets to the active ModelLook"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            self.report({'ERROR'}, "Select an imported model armature or mesh.")
            return {'CANCELLED'}
        sanitize_model_look_metadata(arm, mark_modified=True)
        looks = _model_json_list_for_edit(arm, "engine_model_looks_json")
        if not looks:
            self.report({'ERROR'}, "No looks exist on this model.")
            return {'CANCELLED'}
        subset_ids = _selected_model_subset_ids(context, arm, assign_missing=True)
        if not subset_ids:
            self.report({'ERROR'}, "Select one or more mesh children to add.")
            return {'CANCELLED'}
        look_index = max(0, min(_active_model_look_index(arm), len(looks) - 1))
        existing = [int(value) for value in looks[look_index].get("subset_ids", [])]
        for subset_id in subset_ids:
            if subset_id not in existing:
                existing.append(subset_id)
        _normalize_model_look(arm, looks[look_index], existing)
        custom_ids = _custom_model_subset_ids(arm, subset_ids)
        if custom_ids:
            for other_index, other_look in enumerate(looks):
                if other_index == look_index:
                    continue
                other_ids = [int(value) for value in other_look.get("subset_ids", [])]
                kept = [value for value in other_ids if value not in custom_ids]
                if len(kept) != len(other_ids):
                    _normalize_model_look(arm, other_look, kept)
        _write_model_json_list(arm, "engine_model_looks_json", looks)
        arm["engine_model_looks_modified"] = True
        update_lod_visibility(arm, context)
        self.report({'INFO'}, f"Added {len(subset_ids)} selected subset(s) to Look {look_index}.")
        return {'FINISHED'}


class MODEL_OT_remove_selected_from_look(Operator):
    bl_idname = "model.remove_selected_from_luna_look"
    bl_label = "Remove Selected Meshes"
    bl_description = "Remove selected mesh subsets from the active ModelLook"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            self.report({'ERROR'}, "Select an imported model armature or mesh.")
            return {'CANCELLED'}
        sanitize_model_look_metadata(arm, mark_modified=True)
        looks = _model_json_list_for_edit(arm, "engine_model_looks_json")
        if not looks:
            self.report({'ERROR'}, "No looks exist on this model.")
            return {'CANCELLED'}
        subset_ids = set(_selected_model_subset_ids(context, arm, assign_missing=False))
        if not subset_ids:
            self.report({'ERROR'}, "Select one or more mesh children to remove.")
            return {'CANCELLED'}
        look_index = max(0, min(_active_model_look_index(arm), len(looks) - 1))
        existing = [int(value) for value in looks[look_index].get("subset_ids", [])]
        kept = [value for value in existing if value not in subset_ids]
        _normalize_model_look(arm, looks[look_index], kept)
        _write_model_json_list(arm, "engine_model_looks_json", looks)
        arm["engine_model_looks_modified"] = True
        update_lod_visibility(arm, context)
        self.report({'INFO'}, f"Removed selected subset(s) from Look {look_index}.")
        return {'FINISHED'}


class MODEL_OT_add_selected_to_look_group(Operator):
    bl_idname = "model.add_selected_to_luna_look_group"
    bl_label = "Add Active Look To Group"
    bl_description = "Add the active ModelLook to the active ModelLookGroup"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            self.report({'ERROR'}, "Select an imported model armature or mesh.")
            return {'CANCELLED'}
        sanitize_model_look_metadata(arm, mark_modified=True)
        looks = _model_json_list_for_edit(arm, "engine_model_looks_json")
        groups = _model_json_list_for_edit(arm, "engine_model_look_groups_json")
        if not looks:
            self.report({'ERROR'}, "No looks exist on this model.")
            return {'CANCELLED'}
        if not groups:
            self.report({'ERROR'}, "No look groups exist on this model.")
            return {'CANCELLED'}
        group_index = max(0, min(_active_model_group_index(arm), len(groups) - 1))
        look_index = max(0, min(_active_model_look_index(arm), len(looks) - 1))
        look_indices = []
        for value in groups[group_index].get("look_indices", []) or []:
            try:
                existing_index = int(value)
            except Exception:
                continue
            if 0 <= existing_index < len(looks) and existing_index not in look_indices:
                look_indices.append(existing_index)
        if look_index not in look_indices:
            look_indices.append(look_index)
        groups[group_index]["look_indices"] = look_indices
        _write_model_json_list(arm, "engine_model_look_groups_json", groups)
        arm["engine_model_looks_modified"] = True
        update_lod_visibility(arm, context)
        self.report({'INFO'}, f"Added Look {look_index} to Look Group {group_index}.")
        return {'FINISHED'}


class MODEL_OT_remove_selected_from_look_group(Operator):
    bl_idname = "model.remove_selected_from_luna_look_group"
    bl_label = "Remove Active Look From Group"
    bl_description = "Remove the active ModelLook from the active ModelLookGroup"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _model_armature_from_context(context)
        if not arm:
            self.report({'ERROR'}, "Select an imported model armature or mesh.")
            return {'CANCELLED'}
        sanitize_model_look_metadata(arm, mark_modified=True)
        looks = _model_json_list_for_edit(arm, "engine_model_looks_json")
        groups = _model_json_list_for_edit(arm, "engine_model_look_groups_json")
        if not groups:
            self.report({'ERROR'}, "No look groups exist on this model.")
            return {'CANCELLED'}
        group_index = max(0, min(_active_model_group_index(arm), len(groups) - 1))
        look_index = max(0, min(_active_model_look_index(arm), max(0, len(looks) - 1)))
        kept = []
        removed = False
        for value in groups[group_index].get("look_indices", []):
            try:
                existing_index = int(value)
            except Exception:
                continue
            if not (0 <= existing_index < len(looks)):
                continue
            if existing_index == look_index:
                removed = True
                continue
            if existing_index not in kept:
                kept.append(existing_index)
        groups[group_index]["look_indices"] = kept
        _write_model_json_list(arm, "engine_model_look_groups_json", groups)
        arm["engine_model_looks_modified"] = True
        update_lod_visibility(arm, context)
        if removed:
            self.report({'INFO'}, f"Removed Look {look_index} from Look Group {group_index}.")
        else:
            self.report({'INFO'}, f"Look {look_index} was not in Look Group {group_index}.")
        return {'FINISHED'}

EVENT_CLIPBOARD_KIND = "LUNA_ENGINE_IO_ANIM_EVENT"
EVENT_CLIPBOARD_VERSION = 1

def _event_action_from_context(context):
    arm = _resolve_anim_armature(context)
    action = _resolve_anim_action(arm) if arm else None
    if action:
        return action
    obj = getattr(context, "active_object", None)
    if obj and getattr(obj, "animation_data", None):
        return obj.animation_data.action
    return None

def _next_event_marker_index(action):
    indices = []
    for marker in getattr(action, "pose_markers", []) or []:
        match = re.match(r"Trigger_\[(\d+)\]", marker.name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices) + 1 if indices else 0

def _jsonable_event_prop_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return list(value)
    except Exception:
        return str(value)

def _set_event_prop_value(action, key, value):
    try:
        action[key] = value
    except Exception:
        action[key] = str(value)

def _event_clipboard_text(data):
    return json.dumps(data, separators=(",", ":"), sort_keys=True)

def _parse_event_clipboard_text(text):
    try:
        data = json.loads(str(text or ""))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("kind") != EVENT_CLIPBOARD_KIND:
        return None
    if int(data.get("version", 0)) != EVENT_CLIPBOARD_VERSION:
        return None
    if not isinstance(data.get("props"), dict):
        return None
    return data

def _copied_event_payload_b64(action, marker_index):
    try:
        pt_dict = json.loads(str(action.get("engine_passthrough_blocks", "{}") or "{}"))
        tb_data = base64.b64decode(pt_dict.get(str(BLOCK_HASHES["AnimClipTracksData"]), ""))
        tg_data = base64.b64decode(pt_dict.get(str(BLOCK_HASHES["AnimClipTriggerData"]), ""))
        loc_count, trig_count, ev_size, _marker_count, _tb_data = _read_tracks_counts(tb_data)
    except Exception:
        return ""

    if marker_index < 0 or marker_index >= trig_count:
        return ""
    locator_size = (loc_count + 1) * TRIGGER_LOCATOR_JOINT_STRIDE if loc_count else 0
    records_off = locator_size + int(ev_size)
    record_off = records_off + marker_index * ANIM_TRIGGER_RECORD_SIZE
    if record_off < 0 or record_off + ANIM_TRIGGER_RECORD_SIZE > len(tg_data):
        return ""
    if records_off < 0 or records_off > len(tg_data):
        return ""

    try:
        ev_off = struct.unpack_from("<H", tg_data, record_off + TRIGGER_EVENT_OFFSET_OFFSET)[0] << TRIGGER_EVENT_OFFSET_SHIFT
    except Exception:
        return ""
    if ev_off < locator_size or ev_off + TRIGGER_PAYLOAD_HEADER_SIZE > records_off:
        return ""

    event_offsets = []
    for i in range(trig_count):
        off = records_off + i * ANIM_TRIGGER_RECORD_SIZE
        if off + ANIM_TRIGGER_RECORD_SIZE > len(tg_data):
            break
        try:
            candidate = struct.unpack_from("<H", tg_data, off + TRIGGER_EVENT_OFFSET_OFFSET)[0] << TRIGGER_EVENT_OFFSET_SHIFT
        except Exception:
            continue
        if locator_size <= candidate < records_off:
            event_offsets.append(candidate)
    next_offsets = [candidate for candidate in event_offsets if candidate > ev_off]
    raw_end = min(next_offsets) if next_offsets else records_off
    payload_start = ev_off + TRIGGER_PAYLOAD_HEADER_SIZE
    if payload_start + DATABUFFER_HEADER_SIZE > raw_end:
        return ""

    try:
        _schema_crc, magic, _num_fields, size_object = struct.unpack_from("<IIII", tg_data, payload_start)
    except Exception:
        return ""
    if magic != DATABUFFER_MAGIC:
        return ""
    payload_end = min(payload_start + DATABUFFER_HEADER_SIZE + max(0, int(size_object)), raw_end)
    if payload_end <= payload_start:
        return ""
    return base64.b64encode(tg_data[payload_start:payload_end]).decode("ascii")

class ANIM_OT_copy_event(Operator):
    bl_idname = "anim.copy_event"
    bl_label = "Copy Animation Event"
    bl_description = "Copy this Luna Engine animation event so it can be pasted into another animation"

    marker_name: StringProperty()

    def execute(self, context):
        action = _event_action_from_context(context)
        marker = action.pose_markers.get(self.marker_name) if action else None
        if not marker:
            self.report({'ERROR'}, "Could not find the animation event to copy.")
            return {'CANCELLED'}

        match = re.match(r"Trigger_\[(\d+)\]", marker.name)
        if not match:
            self.report({'ERROR'}, "Only Luna Engine trigger events can be copied.")
            return {'CANCELLED'}

        idx = int(match.group(1))
        prefix = f"marker_{idx}_"
        props = {}
        for key in action.keys():
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            if suffix in {"name_hash", "event_data_off"}:
                continue
            props[suffix] = _jsonable_event_prop_value(action[key])

        ev_hash = _idprop_u32(action.get(f"{prefix}ev_hash", 0), 0)
        event_name = str(action.get(f"{prefix}event_name", "")) or _event_name_for_hash(ev_hash) or f"Unknown_{ev_hash:08X}"
        data = {
            "kind": EVENT_CLIPBOARD_KIND,
            "version": EVENT_CLIPBOARD_VERSION,
            "event_name": event_name,
            "ev_hash": ev_hash,
            "frame": float(marker.frame),
            "props": props,
        }
        event_payload_b64 = _copied_event_payload_b64(action, idx)
        if event_payload_b64:
            data["event_payload_b64"] = event_payload_b64

        context.window_manager.clipboard = _event_clipboard_text(data)
        self.report({'INFO'}, f"Copied {event_name}.")
        return {'FINISHED'}

class ANIM_OT_paste_event(Operator):
    bl_idname = "anim.paste_event"
    bl_label = "Paste Animation Event"
    bl_description = "Paste the copied Luna Engine event into the active animation at the current frame"

    def execute(self, context):
        action = _event_action_from_context(context)
        if not action:
            self.report({'ERROR'}, "Select an armature with an active animation first.")
            return {'CANCELLED'}

        data = _parse_event_clipboard_text(getattr(context.window_manager, "clipboard", ""))
        if not data:
            self.report({'ERROR'}, "The clipboard does not contain a copied Luna Engine animation event.")
            return {'CANCELLED'}

        _ensure_event_passthrough_blocks(action)

        next_idx = _next_event_marker_index(action)
        prefix = f"marker_{next_idx}_"
        ev_hash = _idprop_u32(data.get("ev_hash", 0), 0)
        marker_name = f"Trigger_[{next_idx}]_{ev_hash:08X}"
        marker = action.pose_markers.new(name=marker_name)
        marker.frame = context.scene.frame_current

        action[f"{prefix}name_hash"] = to_signed_32(string_crc32(marker_name))
        for suffix, value in data.get("props", {}).items():
            if suffix in {"name_hash", "event_data_off"}:
                continue
            _set_event_prop_value(action, f"{prefix}{suffix}", value)

        action[f"{prefix}ev_hash"] = to_signed_32(ev_hash)
        if str(data.get("event_name", "") or ""):
            action[f"{prefix}event_name"] = str(data["event_name"])
        if "actor_hash" not in data.get("props", {}):
            action[f"{prefix}actor_hash"] = 0
        if "loc_hash" not in data.get("props", {}):
            action[f"{prefix}loc_hash"] = 0
        if "flags" not in data.get("props", {}):
            action[f"{prefix}flags"] = 0
        if "rad" not in data.get("props", {}):
            action[f"{prefix}rad"] = 0

        event_payload_b64 = str(data.get("event_payload_b64", "") or "")
        if event_payload_b64:
            action[f"{prefix}event_payload_b64"] = event_payload_b64
            action[f"{prefix}rebuild_payload"] = False
        else:
            action[f"{prefix}rebuild_payload"] = True

        event_name = str(action.get(f"{prefix}event_name", "")) or f"0x{ev_hash:08X}"
        self.report({'INFO'}, f"Pasted {event_name} at frame {int(marker.frame)}.")
        return {'FINISHED'}

class ANIM_OT_add_event(Operator):
    bl_idname = "anim.add_event"
    bl_label = "Add Animation Event"
    bl_description = "Add a new Luna Engine animation event at the current frame"

    event_name: StringProperty(
        name="Event Type",
        description="Exact event type selected from the search results",
        default="",
        options={'SKIP_SAVE'},
    )
    event_search: StringProperty(
        name="Event Search",
        description="Type part of an event name; results are filtered instead of opening a giant dropdown",
        default="",
        options={'SKIP_SAVE'},
    )

    def invoke(self, context, event):



        if str(self.event_name or "").strip():
            return self.execute(context)
        if not self.event_search:
            self.event_search = "damage"
        return context.window_manager.invoke_props_dialog(self, width=480)

    def draw(self, context):
        layout = self.layout
        _draw_event_type_search(layout, self.bl_idname, self)

    def execute(self, context):
        obj = context.active_object
        if not obj or not obj.animation_data or not obj.animation_data.action:
            return {'CANCELLED'}
        action = obj.animation_data.action
        _ensure_event_passthrough_blocks(action)

        event_name = str(self.event_name or "").strip() or _event_type_best_match(self.event_search)
        if not event_name:
            self.report({'ERROR'}, "Search is ambiguous. Type more letters or click one of the shown results.")
            return {'CANCELLED'}

        indices = []
        for m in action.pose_markers:
            match = re.match(r"Trigger_\[(\d+)\]", m.name)
            if match:
                indices.append(int(match.group(1)))
        next_idx = max(indices) + 1 if indices else 0
        ev_hash = _event_hash_for_name(event_name)
        marker_name = f"Trigger_[{next_idx}]_{ev_hash:08X}"
        marker = action.pose_markers.new(name=marker_name)
        marker.frame = context.scene.frame_current

        prefix = f"marker_{next_idx}_"
        action[f"{prefix}name_hash"] = to_signed_32(string_crc32(marker_name))
        action[f"{prefix}actor_hash"] = 0
        action[f"{prefix}loc_hash"] = 0
        action[f"{prefix}flags"] = 0
        action[f"{prefix}rad"] = 0
        set_action_event_type(action, prefix, event_name, preserve_existing=False)
        return {'FINISHED'}

class ANIM_OT_change_event_type(Operator):
    bl_idname = "anim.change_event_type"
    bl_label = "Change Animation Event Type"
    bl_description = "Change this event type and rebuild its editable DDL property list"

    marker_name: StringProperty()
    event_name: StringProperty(
        name="Event Type",
        description="Exact event type selected from the search results",
        default="",
        options={'SKIP_SAVE'},
    )
    event_search: StringProperty(
        name="Event Search",
        description="Type part of an event name; results are filtered instead of opening a giant dropdown",
        default="",
        options={'SKIP_SAVE'},
    )

    def invoke(self, context, event):



        if str(self.event_name or "").strip():
            return self.execute(context)
        obj = context.active_object
        action = obj.animation_data.action if obj and obj.animation_data else None
        marker = action.pose_markers.get(self.marker_name) if action else None
        current_name = ""
        if marker:
            match = re.match(r"Trigger_\[(\d+)\]", marker.name)
            if match:
                prefix = f"marker_{match.group(1)}_"
                ev_hash = _idprop_u32(action.get(f"{prefix}ev_hash", 0))
                current_name = str(action.get(f"{prefix}event_name", "")) or _event_name_for_hash(ev_hash)
        if not self.event_search:
            self.event_search = current_name or "damage"
        return context.window_manager.invoke_props_dialog(self, width=480)

    def draw(self, context):
        layout = self.layout
        selected_label = ""
        obj = context.active_object
        action = obj.animation_data.action if obj and obj.animation_data else None
        marker = action.pose_markers.get(self.marker_name) if action else None
        if marker:
            match = re.match(r"Trigger_\[(\d+)\]", marker.name)
            if match:
                prefix = f"marker_{match.group(1)}_"
                selected_label = str(action.get(f"{prefix}event_name", "")) or _event_name_for_hash(_idprop_u32(action.get(f"{prefix}ev_hash", 0)))
        _draw_event_type_search(layout, self.bl_idname, self, marker_name=self.marker_name, selected_label=selected_label)

    def execute(self, context):
        obj = context.active_object
        action = obj.animation_data.action if obj and obj.animation_data else None
        if not action:
            return {'CANCELLED'}
        marker = action.pose_markers.get(self.marker_name)
        if not marker:
            return {'CANCELLED'}
        match = re.match(r"Trigger_\[(\d+)\]", marker.name)
        if not match:
            return {'CANCELLED'}

        event_name = str(self.event_name or "").strip() or _event_type_best_match(self.event_search)
        if not event_name:
            self.report({'ERROR'}, "Search is ambiguous. Type more letters or click one of the shown results.")
            return {'CANCELLED'}

        idx = match.group(1)
        prefix = f"marker_{idx}_"
        ev_hash = _event_hash_for_name(event_name)
        marker.name = f"Trigger_[{idx}]_{ev_hash:08X}"
        action[f"{prefix}name_hash"] = to_signed_32(string_crc32(marker.name))
        set_action_event_type(action, prefix, event_name, preserve_existing=True)
        return {'FINISHED'}

def _ensure_event_passthrough_blocks(action):
    if "engine_passthrough_blocks" not in action:
        pt = {}
    else:
        try:
            pt = json.loads(action["engine_passthrough_blocks"])
        except Exception:
            pt = {}
    if str(BLOCK_HASHES["AnimClipTracksData"]) not in pt:
        pt[str(BLOCK_HASHES["AnimClipTracksData"])] = base64.b64encode(_empty_tracks_data_block()).decode('ascii')
    if str(BLOCK_HASHES["AnimClipTriggerData"]) not in pt:
        pt[str(BLOCK_HASHES["AnimClipTriggerData"])] = base64.b64encode(b"").decode('ascii')
    action["engine_passthrough_blocks"] = json.dumps(pt)

class ANIM_OT_delete_event(Operator):
    bl_idname = "anim.delete_event"
    bl_label = "Delete Animation Event"
    bl_description = "Delete this Luna Engine animation event"
    
    marker_name: StringProperty()

    def execute(self, context):
        obj = context.active_object
        action = obj.animation_data.action
        
        marker = action.pose_markers.get(self.marker_name)
        if not marker:
            return {'CANCELLED'}
        
        match = re.match(r"Trigger_\[(\d+)\]", marker.name)
        if match:
            idx = match.group(1)
            prefix = f"marker_{idx}_"
            

            keys_to_del = [k for k in action.keys() if k.startswith(prefix)]
            for k in keys_to_del:
                del action[k]
            


        action.pose_markers.remove(marker)
        return {'FINISHED'}
