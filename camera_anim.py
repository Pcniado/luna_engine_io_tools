# This module is part of the public-release split of blender_import_model_anim_release.py.
# Camera AnimClips store their camera transform in AnimClipMotionData and camera
# parameters in custom tracks instead of skeletal joint sample blocks.

from .utils import *

ANIM_CURVE_BUILT_FORMAT = "<IIQIBBBBB7sHHHH"
ANIM_CURVE_FLAG_CATMULL = 0x01
ANIM_CURVE_FLAG_BEZIER = 0x02
CAMERA_TRACK_XFOV = "XFOV"
CAMERA_TRACK_FOCAL_LENGTH = "igFocalLength"
CAMERA_DEFAULT_WRITER_NAME = "topology_file_writer"
CAMERA_CLIP_TYPE = "camera"


def _camera_native_basis_quat():
    quat = SWIZZLE_MAT.to_quaternion()
    quat.normalize()
    return quat


def _engine_camera_to_blender_transform(loc_eng, quat_eng):
    loc_bl, quat_bl = engine_motion_to_blender_transform(loc_eng, quat_eng)
    # Native Blender cameras are not swizzled mesh data. Their local camera
    # axes need the source Y-up camera basis applied after the world conversion.
    quat_bl = quat_bl @ _camera_native_basis_quat()
    quat_bl.normalize()
    return loc_bl, quat_bl


def _blender_camera_to_engine_transform(loc_bl, quat_bl):
    quat_generic = quat_bl @ _camera_native_basis_quat().inverted()
    quat_generic.normalize()
    return blender_transform_to_engine_motion(loc_bl, quat_generic)


def _anim_curve_built_size():
    return struct.calcsize(ANIM_CURVE_BUILT_FORMAT)


def _resolve_anim_camera(context):
    candidates = []
    for attr in ("object", "active_object"):
        obj = getattr(context, attr, None)
        if obj and obj not in candidates:
            candidates.append(obj)
    for obj in getattr(context, "selected_objects", []) or []:
        if obj and obj not in candidates:
            candidates.append(obj)
    for obj in candidates:
        if getattr(obj, "type", None) == "CAMERA":
            return obj
    for obj in candidates:
        for child in getattr(obj, "children", []) or []:
            if getattr(child, "type", None) == "CAMERA":
                return child
    return None


def _ensure_camera_object(context, clip_name):
    camera = _resolve_anim_camera(context)
    if camera:
        _activate_camera_object(context, camera)
        return camera

    data_name = _unique_object_name(f"{clip_name}_CameraData")
    obj_name = _unique_object_name(f"{clip_name}_Camera")
    cam_data = bpy.data.cameras.new(data_name)
    camera = bpy.data.objects.new(obj_name, cam_data)
    collection = getattr(context, "collection", None) or getattr(context.scene, "collection", None)
    if collection:
        collection.objects.link(camera)
    else:
        bpy.context.scene.collection.objects.link(camera)
    try:
        context.view_layer.objects.active = camera
        camera.select_set(True)
    except Exception:
        pass
    return camera


def _activate_camera_object(context, camera):
    if not camera:
        return
    try:
        context.view_layer.objects.active = camera
        camera.select_set(True)
    except Exception:
        pass


def _ensure_id_action(datablock, action_name):
    if not getattr(datablock, "animation_data", None):
        datablock.animation_data_create()
    action = datablock.animation_data.action
    if action is None:
        action = bpy.data.actions.new(name=action_name)
        datablock.animation_data.action = action
    return action


def _write_scalar_fcurve(action, datablock, data_path, frames, values, index=0):
    if not values:
        return None
    fc = _ensure_action_fcurve(action, datablock, data_path, index)
    count = min(len(frames), len(values))
    fc.keyframe_points.add(count)
    co = np.empty(count * 2, dtype=np.float32)
    co[0::2] = np.asarray(frames[:count], dtype=np.float32)
    co[1::2] = np.asarray(values[:count], dtype=np.float32)
    fc.keyframe_points.foreach_set("co", co)
    try:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    except Exception:
        pass
    fc.update()
    return fc


def _write_transform_fcurves(action, obj, frames, samples):
    obj.rotation_mode = "QUATERNION"
    loc_values = [[loc.x, loc.y, loc.z] for loc, _quat in samples]
    rot_values = [[quat.w, quat.x, quat.y, quat.z] for _loc, quat in samples]
    for axis in range(3):
        _write_scalar_fcurve(action, obj, "location", frames, [row[axis] for row in loc_values], axis)
    for axis in range(4):
        _write_scalar_fcurve(action, obj, "rotation_quaternion", frames, [row[axis] for row in rot_values], axis)


def _string_at_from_buffer(buffer, offset):
    if not buffer or offset is None:
        return ""
    offset = int(offset)
    if offset < 0 or offset >= len(buffer):
        return ""
    end = buffer.find(b"\x00", offset)
    if end < 0:
        end = len(buffer)
    raw = buffer[offset:end]
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _dat1_string_table(data, blocks, table_end):
    if data is None or table_end is None:
        return b""
    first_block = min((off for off, _size in blocks.values()), default=len(data))
    if first_block < table_end:
        return b""
    return bytes(data[table_end:first_block])


def _dat1_string_at(data, blocks, table_end, offset):
    return _string_at_from_buffer(_dat1_string_table(data, blocks, table_end), offset)


def _build_string_table(strings):
    data = bytearray()
    offsets = {}

    def add(text):
        text = str(text or "")
        if text in offsets:
            return offsets[text]
        offsets[text] = len(data)
        data.extend(text.encode("utf-8", errors="replace"))
        data.append(0)
        return offsets[text]

    add(CAMERA_DEFAULT_WRITER_NAME)
    for text in strings:
        add(text)
    return bytes(data), offsets


def _curve_record_from_bytes(payload, offset):
    (
        name_hash,
        name_offset,
        tuid,
        key_data_offset,
        dimension,
        flags,
        rotate_order,
        extrapolate_in,
        extrapolate_out,
        _pad,
        key_count_0,
        key_count_1,
        key_count_2,
        key_count_3,
    ) = struct.unpack_from(ANIM_CURVE_BUILT_FORMAT, payload, offset)
    return {
        "name_hash": int(name_hash) & U32_MASK,
        "name_offset": int(name_offset) & U32_MASK,
        "tuid": int(tuid) & U64_MASK,
        "key_data_offset": int(key_data_offset) & U32_MASK,
        "dimension": int(dimension),
        "flags": int(flags) & 0xFF,
        "rotate_order": int(rotate_order) & 0xFF,
        "extrapolate_in": int(extrapolate_in) & 0xFF,
        "extrapolate_out": int(extrapolate_out) & 0xFF,
        "key_counts": [int(key_count_0), int(key_count_1), int(key_count_2), int(key_count_3)],
    }


def _parse_custom_tracks_payload(tracks_payload, data_payload, string_table=b""):
    records = []
    if not tracks_payload or len(tracks_payload) < _anim_curve_built_size():
        return records
    if not data_payload:
        data_payload = b""
    record_size = _anim_curve_built_size()
    count = len(tracks_payload) // record_size
    for index in range(count):
        record = _curve_record_from_bytes(tracks_payload, index * record_size)
        name = _string_at_from_buffer(string_table, record["name_offset"])
        if not name:
            name = f"Track_{record['name_hash']:08X}"
        record["name"] = name
        record["values"] = []
        record["raw_key_data"] = b""
        if record["flags"] & ANIM_CURVE_FLAG_CATMULL:
            dimension = max(1, min(4, int(record["dimension"])))
            key_count = max(0, int(record["key_counts"][0]))
            value_count = key_count * dimension
            key_offset = int(record["key_data_offset"])
            byte_count = value_count * 4
            if key_offset <= len(data_payload) and byte_count <= len(data_payload) - key_offset:
                record["values"] = list(struct.unpack_from(f"<{value_count}f", data_payload, key_offset)) if value_count else []
                record["raw_key_data"] = data_payload[key_offset:key_offset + byte_count]
        else:
            key_offset = int(record["key_data_offset"])
            if key_offset < len(data_payload):
                record["raw_key_data"] = data_payload[key_offset:]
        records.append(record)
    return records


def _parse_custom_tracks_from_blocks(data, blocks, table_end):
    tracks_block = blocks.get(BLOCK_HASHES["AnimClipCustomTracks"])
    data_block = blocks.get(BLOCK_HASHES["AnimClipCustomTrackData"])
    if not tracks_block or not data_block:
        return []
    tracks_off, tracks_size = tracks_block
    data_off, data_size = data_block
    return _parse_custom_tracks_payload(
        data[tracks_off:tracks_off + tracks_size],
        data[data_off:data_off + data_size],
        _dat1_string_table(data, blocks, table_end),
    )


def _track_values_by_name(records):
    values = {}
    for record in records:
        if record.get("values"):
            values[str(record.get("name") or "")] = list(record["values"])
    return values


def looks_like_camera_animclip(data, blocks, table_end, cb):
    clip_flags = int(cb[CLIP_BUILT_FLAGS_INDEX]) & U32_MASK
    if clip_flags & FLAG_CAMERA:
        return True

    # Some camera clips ship without kFlagsCamera set. They still have no
    # joints or skeletal sample blocks, carry motion data, and expose camera
    # parameter curves such as XFOV/igFocalLength.
    if int(cb[CLIP_BUILT_JOINT_COUNT_INDEX]) != 0:
        return False
    if BLOCK_HASHES["AnimClipMotionData"] not in blocks:
        return False

    skeletal_blocks = (
        "AnimClipBaseState",
        "AnimClipSampleElem",
        "AnimClipSampleDataResident",
        "AnimClipSampleDataPaged",
        "AnimClipJointHashes",
    )
    if any(BLOCK_HASHES[name] in blocks for name in skeletal_blocks):
        return False

    records = _parse_custom_tracks_from_blocks(data, blocks, table_end)
    track_names = {str(record.get("name") or "") for record in records}
    if CAMERA_TRACK_XFOV in track_names or CAMERA_TRACK_FOCAL_LENGTH in track_names:
        return True

    return False


def _camera_sensor_width_from_tracks(records):
    values = _track_values_by_name(records)
    xfov = values.get(CAMERA_TRACK_XFOV)
    lens = values.get(CAMERA_TRACK_FOCAL_LENGTH)
    if not xfov or not lens:
        return 0.0
    widths = []
    for fov_value, lens_value in zip(xfov, lens):
        try:
            fov = float(fov_value)
            focal = float(lens_value)
        except Exception:
            continue
        if fov > 0.0 and focal > 0.0:
            widths.append(2.0 * focal * math.tan(fov * 0.5))
    if not widths:
        return 0.0
    widths.sort()
    return float(widths[len(widths) // 2])


def _expanded_track_values(record, sample_count):
    dimension = max(1, int(record.get("dimension", 1)))
    values = list(record.get("values") or [])
    key_count = int(record.get("key_counts", [0])[0])
    if not values:
        return []
    if key_count == 1:
        return values[:dimension] * max(1, sample_count)
    expected = key_count * dimension
    if len(values) >= expected:
        return values[:expected]
    while len(values) < expected:
        values.append(values[-dimension] if len(values) >= dimension else 0.0)
    return values


def _set_camera_clip_metadata(targets, cb, clip_flags, sample_count, fps, duration, clip_name_hash):
    for target in [t for t in targets if t is not None]:
        target["engine_clip_type"] = CAMERA_CLIP_TYPE
        target["engine_clip_flags"] = to_signed_32(clip_flags)
        set_original_flags(target, clip_flags)
        target["engine_clip_name_hash"] = to_signed_32(clip_name_hash)
        target["engine_clip_joint_count"] = 0
        target["engine_sample_cnt"] = int(sample_count)
        target["engine_clip_full_sample_count"] = int(sample_count)
        target["engine_clip_full_fps"] = float(fps)
        target["engine_clip_duration"] = float(duration)
        target["engine_clip_resident_sample_count"] = int(cb[CLIP_BUILT_RESIDENT_SAMPLE_COUNT_INDEX]) if "CLIP_BUILT_RESIDENT_SAMPLE_COUNT_INDEX" in globals() else int(sample_count)
        target["engine_clip_resident_fps"] = float(cb[CLIP_BUILT_RESIDENT_FPS_INDEX]) if "CLIP_BUILT_RESIDENT_FPS_INDEX" in globals() else float(fps)
        target["engine_clip_import_sample_count"] = int(sample_count)
        target["engine_clip_import_fps"] = float(fps)
        target["engine_clip_import_sample_source"] = "camera_motion"
        for prop_name, prop_value in (
            ("engine_clip_cycle_count", cb[CLIP_BUILT_CYCLE_COUNT_INDEX]),
            ("engine_clip_facial_name_hash", cb[CLIP_BUILT_FACIAL_NAME_HASH_INDEX]),
            ("engine_clip_sound_event_hash", cb[CLIP_BUILT_SOUND_EVENT_HASH_INDEX]),
            ("engine_clip_particle_count_max", cb[CLIP_BUILT_PARTICLE_COUNT_MAX_INDEX]),
        ):
            target[prop_name] = to_signed_32(prop_value)


def _luna_settings_state_globals():
    for name in ("store_scene_luna_settings_for_target", "_store_active_luna_settings", "update_anim_fps"):
        func = globals().get(name)
        func_globals = getattr(func, "__globals__", None)
        if isinstance(func_globals, dict) and "_LUNA_SETTINGS_SYNCING" in func_globals:
            return func_globals
    return globals()


def _set_scene_attr_without_luna_store(scene, name, value):
    state_globals = _luna_settings_state_globals()
    had_sync_flag = "_LUNA_SETTINGS_SYNCING" in state_globals
    old_syncing = bool(state_globals.get("_LUNA_SETTINGS_SYNCING", False))
    if had_sync_flag:
        state_globals["_LUNA_SETTINGS_SYNCING"] = True
    try:
        setattr(scene, name, value)
    finally:
        if had_sync_flag:
            state_globals["_LUNA_SETTINGS_SYNCING"] = old_syncing


def _sync_camera_flag_toggles_without_luna_store(scene, clip_flags):
    if "sync_scene_flag_toggles_from_flags" not in globals():
        return
    state_globals = _luna_settings_state_globals()
    had_sync_flag = "_LUNA_SETTINGS_SYNCING" in state_globals
    old_syncing = bool(state_globals.get("_LUNA_SETTINGS_SYNCING", False))
    if had_sync_flag:
        state_globals["_LUNA_SETTINGS_SYNCING"] = True
    try:
        sync_scene_flag_toggles_from_flags(scene, clip_flags)
    finally:
        if had_sync_flag:
            state_globals["_LUNA_SETTINGS_SYNCING"] = old_syncing


def _apply_camera_import_scene_settings(context, camera, fps, sample_count, clip_flags):
    scene = context.scene
    settings = {
        "use_smooth_playback": False,
        "engine_anim_fps": float(fps),
        "engine_export_frame_start": 0,
        "engine_export_frame_end": max(0, int(sample_count) - 1),
        "engine_export_use_original_values": True,
        "engine_export_looping": bool(int(clip_flags) & FLAG_LOOPING),
        "engine_export_additive": bool(int(clip_flags) & FLAG_IS_ADDITIVE),
        "engine_export_partial": bool(int(clip_flags) & FLAG_IS_PARTIAL),
        "engine_export_partial_motion": bool(int(clip_flags) & FLAG_PARTIAL_MOTION),
    }
    for prop_name, prop_value in settings.items():
        if hasattr(scene, prop_name):
            _set_scene_attr_without_luna_store(scene, prop_name, prop_value)

    scene["engine_base_fps"] = float(fps)
    scene.render.frame_map_old = 100
    scene.render.frame_map_new = 100
    scene.frame_start, scene.frame_end = 0, max(0, int(sample_count) - 1)
    _sync_camera_flag_toggles_without_luna_store(scene, clip_flags)

    if "store_scene_luna_settings_for_target" in globals():
        store_scene_luna_settings_for_target(scene, camera)
    if "mark_luna_settings_target" in globals():
        mark_luna_settings_target(scene, camera)


def _motion_header_initial_matrix(header):
    rx, ry, rz, rw = header.get("initial_rot0", (0.0, 0.0, 0.0, 1.0))
    loc = mathutils.Vector(header.get("initial_trans0", (0.0, 0.0, 0.0)))
    quat = mathutils.Quaternion((rw, rx, ry, rz))
    quat.normalize()
    return _matrix_from_loc_quat(loc, quat)


def _camera_samples_from_motion(motion_block, sample_count):
    header = _read_motion_header(motion_block)
    sample_count = max(1, int(sample_count))
    if not motion_block:
        loc = mathutils.Vector((0.0, 0.0, 0.0))
        quat = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
        return [(loc, quat) for _ in range(sample_count)], header

    base_matrix = _motion_header_initial_matrix(header)
    supported = bool(header.get("supported"))
    motion_count = int(header.get("sample_count", 0))
    if not supported or motion_count <= 0:
        loc_eng, quat_eng = _loc_quat_from_matrix(base_matrix)
        loc_bl, quat_bl = _engine_camera_to_blender_transform(loc_eng, quat_eng)
        return [(loc_bl.copy(), quat_bl.copy()) for _ in range(sample_count)], header

    samples = []
    count = max(1, motion_count)
    for i in range(count):
        off = int(header["sample_start"]) + (i * ANIM_MOTION_STANDARD_SAMPLE_SIZE)
        rel_loc, rel_quat, _phase = _decode_standard_motion_sample(motion_block, off)
        rel_matrix = _matrix_from_loc_quat(rel_loc, rel_quat)
        loc_eng, quat_eng = _loc_quat_from_matrix(base_matrix @ rel_matrix)
        loc_bl, quat_bl = _engine_camera_to_blender_transform(loc_eng, quat_eng)
        samples.append((loc_bl, quat_bl))
    while len(samples) < sample_count:
        loc, quat = samples[-1]
        samples.append((loc.copy(), quat.copy()))
    return samples[:sample_count], header


def _store_camera_passthrough(action, data, blocks, handled_hashes):
    passthrough = {}
    for block_hash, (block_off, block_size) in blocks.items():
        if block_hash in handled_hashes:
            continue
        passthrough[str(block_hash)] = base64.b64encode(data[block_off:block_off + block_size]).decode("ascii")
    if passthrough:
        action["engine_passthrough_blocks"] = json.dumps(passthrough)


def import_camera_animclip(context, operator, data, blocks, table_end, cb, filepath, wm=None):
    clip_flags = int(cb[CLIP_BUILT_FLAGS_INDEX]) & U32_MASK
    clip_name_hash = int(cb[CLIP_BUILT_NAME_HASH_INDEX]) & U32_MASK
    fps = float(cb[CLIP_BUILT_PAGED_FPS_INDEX]) if float(cb[CLIP_BUILT_PAGED_FPS_INDEX]) > 0.0 else DEFAULT_IMPORT_FPS
    duration = float(cb[CLIP_BUILT_DURATION_INDEX])

    motion_block_ref = blocks.get(BLOCK_HASHES["AnimClipMotionData"])
    motion_block = b""
    motion_sample_count = 0
    if motion_block_ref:
        motion_off, motion_size = motion_block_ref
        motion_block = data[motion_off:motion_off + motion_size]
        motion_header = _read_motion_header(motion_block)
        motion_sample_count = int(motion_header.get("sample_count", 0))

    sample_count = max(1, motion_sample_count or int(cb[CLIP_BUILT_PAGED_SAMPLE_COUNT_INDEX]) or int(cb[CLIP_BUILT_RESIDENT_SAMPLE_COUNT_INDEX]) or 1)
    clip_name = _dat1_string_at(data, blocks, table_end, int(cb[0]))
    if not clip_name:
        clip_name = os.path.splitext(os.path.basename(filepath))[0]

    if wm:
        wm.progress_update(5)

    camera = _ensure_camera_object(context, clip_name)
    action = bpy.data.actions.new(name=os.path.basename(filepath))
    camera.animation_data_create()
    camera.animation_data.action = action
    camera_binding_id = _new_clip_binding_id() if "_new_clip_binding_id" in globals() else None
    if "_ensure_clip_binding_id" in globals():
        _ensure_clip_binding_id(camera, action, camera.data if getattr(camera, "data", None) else None, binding_id=camera_binding_id)
    camera["engine_clip_type"] = CAMERA_CLIP_TYPE

    _activate_camera_object(context, camera)
    _apply_camera_import_scene_settings(context, camera, fps, sample_count, clip_flags)

    frames = list(range(sample_count))
    samples, motion_header = _camera_samples_from_motion(motion_block, sample_count)
    _write_transform_fcurves(action, camera, frames, samples)
    if wm:
        wm.progress_update(55)

    custom_records = _parse_custom_tracks_from_blocks(data, blocks, table_end)
    sensor_width = _camera_sensor_width_from_tracks(custom_records)
    if sensor_width > 0.0 and getattr(camera, "data", None):
        camera.data.sensor_width = sensor_width
        camera.data["engine_camera_sensor_width"] = float(sensor_width)

    camera_data_action = None
    track_values = _track_values_by_name(custom_records)
    if getattr(camera, "data", None) and track_values:
        camera_data_action = _ensure_id_action(camera.data, f"{os.path.basename(filepath)} Camera Data")
        if "_ensure_clip_binding_id" in globals():
            _ensure_clip_binding_id(camera, action, camera.data, camera_data_action, binding_id=_clip_binding_id(camera, action))
        if CAMERA_TRACK_FOCAL_LENGTH in track_values:
            camera.data.lens = float(track_values[CAMERA_TRACK_FOCAL_LENGTH][0])
            _write_scalar_fcurve(camera_data_action, camera.data, "lens", frames, track_values[CAMERA_TRACK_FOCAL_LENGTH])
        elif CAMERA_TRACK_XFOV in track_values:
            camera.data.angle_x = float(track_values[CAMERA_TRACK_XFOV][0])
            _write_scalar_fcurve(camera_data_action, camera.data, "angle_x", frames, track_values[CAMERA_TRACK_XFOV])
        if CAMERA_TRACK_XFOV in track_values:
            camera.data["engine_camera_xfov_first"] = float(track_values[CAMERA_TRACK_XFOV][0])

    string_table = _dat1_string_table(data, blocks, table_end)
    if string_table:
        action["engine_string_table_blob"] = base64.b64encode(string_table).decode("ascii")
    if custom_records:
        action["engine_camera_custom_tracks"] = json.dumps([
            {
                "name": r.get("name", ""),
                "name_hash": to_signed_32(r.get("name_hash", 0)),
                "dimension": int(r.get("dimension", 0)),
                "flags": int(r.get("flags", 0)),
                "key_counts": list(r.get("key_counts", [])),
            }
            for r in custom_records
        ])

    _set_camera_clip_metadata((camera, action, camera.data if getattr(camera, "data", None) else None), cb, clip_flags, sample_count, fps, duration, clip_name_hash)
    _store_motion_metadata(camera, action, motion_block, bool(motion_header.get("supported")), motion_header.get("reason", ""), int(motion_header.get("sample_count", 0)), 0, max(0, sample_count - 1))
    if camera_data_action:
        _set_camera_clip_metadata((camera_data_action,), cb, clip_flags, sample_count, fps, duration, clip_name_hash)
    if "store_scene_luna_settings_for_target" in globals():
        store_scene_luna_settings_for_target(context.scene, camera)
    if "mark_luna_settings_target" in globals():
        mark_luna_settings_target(context.scene, camera)

    handled = {
        BLOCK_HASHES["AnimClipBuilt"],
    }
    _store_camera_passthrough(action, data, blocks, handled)

    if wm:
        wm.progress_update(100)
    operator.report({"INFO"}, f"Imported camera AnimClip: {sample_count} samples @ {fps:.2f} FPS")
    return {"FINISHED"}


def _sample_camera_world(context, camera, frames):
    scene = context.scene
    samples = []
    for frame in frames:
        scene.frame_set(int(round(frame)))
        loc, quat, _scale = camera.matrix_world.decompose()
        quat.normalize()
        samples.append((loc.copy(), quat.copy()))
    return samples


def _sample_camera_scalar(context, camera, frames, prop_name, fallback=0.0):
    scene = context.scene
    values = []
    cam_data = getattr(camera, "data", None)
    for frame in frames:
        scene.frame_set(int(round(frame)))
        try:
            values.append(float(getattr(cam_data, prop_name)))
        except Exception:
            values.append(float(fallback))
    return values


def _build_camera_motion_block(samples_bl, template_block=None, is_looping=False):
    if not samples_bl:
        identity_loc = mathutils.Vector((0.0, 0.0, 0.0))
        identity_quat = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
        samples_bl = [(identity_loc, identity_quat)]

    engine_mats = []
    abs_engine_locs = []
    for loc_bl, quat_bl in samples_bl:
        loc_eng, quat_eng = _blender_camera_to_engine_transform(loc_bl, quat_bl)
        engine_mats.append(_matrix_from_loc_quat(loc_eng, quat_eng))
        abs_engine_locs.append(loc_eng.copy())

    if is_looping and engine_mats:
        engine_mats.append(engine_mats[0].copy())
        abs_engine_locs.append(abs_engine_locs[0].copy())

    base_mat = engine_mats[0].copy()
    base_inv = base_mat.inverted()
    base_loc, base_quat = _loc_quat_from_matrix(base_mat)

    rel_samples = []
    for mat in engine_mats:
        rel_samples.append(_loc_quat_from_matrix(base_inv @ mat))

    sample_bytes = bytearray()
    previous_quat = None
    for loc, quat in rel_samples:
        if previous_quat is not None and previous_quat.dot(quat) < 0.0:
            quat.negate()
        previous_quat = quat.copy()
        sample_bytes.extend(_pack_standard_motion_sample(loc, quat, 0))
    while (ANIM_MOTION_HEADER_SIZE + len(sample_bytes)) % ANIM_MOTION_BLOCK_ALIGN:
        sample_bytes.append(0)

    template_header = _read_motion_header(template_block) if template_block else {}
    if template_header.get("present"):
        block = bytearray((template_block or b"")[:ANIM_MOTION_HEADER_SIZE])
        if len(block) < ANIM_MOTION_HEADER_SIZE:
            block.extend(b"\x00" * (ANIM_MOTION_HEADER_SIZE - len(block)))
    else:
        block = bytearray(ANIM_MOTION_HEADER_SIZE)
        struct.pack_into("<ffff", block, MOTION_INITIAL_ROT1_OFFSET, 0.0, 0.0, 0.0, 1.0)
        struct.pack_into("<fff", block, MOTION_INITIAL_TRANS1_OFFSET, 0.0, 0.0, 0.0)

    struct.pack_into("<ffff", block, MOTION_INITIAL_ROT0_OFFSET, base_quat.x, base_quat.y, base_quat.z, base_quat.w)
    struct.pack_into("<fff", block, MOTION_INITIAL_TRANS0_OFFSET, base_loc.x, base_loc.y, base_loc.z)
    motion_sample_count = len(rel_samples)
    phase_blend = int(template_header.get("phase_blend_samples", 0) or 0)
    if phase_blend <= 0 or phase_blend > motion_sample_count:
        phase_blend = motion_sample_count
    struct.pack_into("<BBH", block, MOTION_TYPE_OFFSET, ANIM_MOTION_TYPE_STANDARD, ANIM_MOTION_FLAG_HAS_MOTION, motion_sample_count)
    struct.pack_into("<IIHB5s", block, MOTION_STREAM_HEADER_OFFSET, 0, len(sample_bytes), phase_blend, 0, b"\x00" * MOTION_HEADER_PAD_BYTES)

    min_v = mathutils.Vector((min(v.x for v in abs_engine_locs), min(v.y for v in abs_engine_locs), min(v.z for v in abs_engine_locs)))
    max_v = mathutils.Vector((max(v.x for v in abs_engine_locs), max(v.y for v in abs_engine_locs), max(v.z for v in abs_engine_locs)))
    struct.pack_into("<fff", block, MOTION_BOUNDS_MIN_OFFSET, min_v.x, min_v.y, min_v.z)
    struct.pack_into("<fff", block, MOTION_BOUNDS_MAX_OFFSET, max_v.x, max_v.y, max_v.z)
    return bytes(block) + bytes(sample_bytes)


def _existing_camera_custom_records(action):
    if not action or "engine_passthrough_blocks" not in action:
        return [], b"", b""
    try:
        pt = json.loads(str(action["engine_passthrough_blocks"]))
        tracks_payload = base64.b64decode(pt.get(str(BLOCK_HASHES["AnimClipCustomTracks"]), ""))
        data_payload = base64.b64decode(pt.get(str(BLOCK_HASHES["AnimClipCustomTrackData"]), ""))
        string_table = base64.b64decode(str(action.get("engine_string_table_blob", ""))) if action.get("engine_string_table_blob") else b""
    except Exception:
        return [], b"", b""
    return _parse_custom_tracks_payload(tracks_payload, data_payload, string_table), tracks_payload, data_payload


def _camera_custom_track_specs(action):
    records, _tracks_payload, _data_payload = _existing_camera_custom_records(action)
    by_name = {str(r.get("name") or ""): r for r in records}
    specs = []
    for name in (CAMERA_TRACK_XFOV, CAMERA_TRACK_FOCAL_LENGTH):
        existing = by_name.get(name)
        if existing:
            specs.append(existing)
        else:
            specs.append({
                "name": name,
                "name_hash": string_crc32(name),
                "name_offset": 0,
                "tuid": 0,
                "dimension": 1,
                "flags": ANIM_CURVE_FLAG_CATMULL,
                "rotate_order": 0,
                "extrapolate_in": 0,
                "extrapolate_out": 0,
                "key_counts": [0, 0, 0, 0],
            })
    for record in records:
        if str(record.get("name") or "") not in {CAMERA_TRACK_XFOV, CAMERA_TRACK_FOCAL_LENGTH}:
            specs.append(record)
    specs.sort(key=lambda r: int(r.get("name_hash", 0)) & U32_MASK)
    return specs


def _build_camera_custom_track_blocks(context, camera, action, frames, string_offsets):
    specs = _camera_custom_track_specs(action)
    existing_values = {str(r.get("name") or ""): _expanded_track_values(r, len(frames)) for r in specs}
    xfov_values = _sample_camera_scalar(context, camera, frames, "angle_x", fallback=0.785398)
    lens_values = _sample_camera_scalar(context, camera, frames, "lens", fallback=50.0)
    values_by_name = {
        CAMERA_TRACK_XFOV: xfov_values,
        CAMERA_TRACK_FOCAL_LENGTH: lens_values,
    }

    track_payload = bytearray()
    data_payload = bytearray()
    sampled_count = 0
    non_sampled_count = 0
    for spec in specs:
        name = str(spec.get("name") or "")
        dimension = max(1, min(4, int(spec.get("dimension", 1) or 1)))
        flags = int(spec.get("flags", ANIM_CURVE_FLAG_CATMULL)) & 0xFF
        data_payload.extend(b"\x00" * ((16 - (len(data_payload) % 16)) % 16))
        key_offset = len(data_payload)

        if flags & ANIM_CURVE_FLAG_CATMULL:
            values = values_by_name.get(name)
            if values is None:
                values = existing_values.get(name, [])
            if not values:
                values = [0.0] * (len(frames) * dimension)
            if dimension == 1:
                packed_values = [float(v) for v in values[:len(frames)]]
            else:
                expected = len(frames) * dimension
                packed_values = [float(v) for v in values[:expected]]
                while len(packed_values) < expected:
                    packed_values.append(0.0)
            data_payload.extend(struct.pack(f"<{len(packed_values)}f", *packed_values))
            key_counts = [len(frames), 0, 0, 0]
            sampled_count += 1
        elif flags & ANIM_CURVE_FLAG_BEZIER:
            raw = bytes(spec.get("raw_key_data") or b"")
            data_payload.extend(raw)
            key_counts = list(spec.get("key_counts", [0, 0, 0, 0]))[:4]
            non_sampled_count += 1
        else:
            values = existing_values.get(name, [])
            if values:
                data_payload.extend(struct.pack(f"<{len(values)}f", *[float(v) for v in values]))
            key_counts = list(spec.get("key_counts", [0, 0, 0, 0]))[:4]
            non_sampled_count += 1
        while len(key_counts) < 4:
            key_counts.append(0)

        name_hash = int(spec.get("name_hash", string_crc32(name))) & U32_MASK
        name_offset = int(string_offsets.get(name, 0)) & U32_MASK
        track_payload.extend(struct.pack(
            ANIM_CURVE_BUILT_FORMAT,
            name_hash,
            name_offset,
            int(spec.get("tuid", 0)) & U64_MASK,
            key_offset,
            dimension,
            flags,
            int(spec.get("rotate_order", 0)) & 0xFF,
            int(spec.get("extrapolate_in", 0)) & 0xFF,
            int(spec.get("extrapolate_out", 0)) & 0xFF,
            b"\x00" * 7,
            int(key_counts[0]) & 0xFFFF,
            int(key_counts[1]) & 0xFFFF,
            int(key_counts[2]) & 0xFFFF,
            int(key_counts[3]) & 0xFFFF,
        ))

    return bytes(track_payload), bytes(data_payload), sampled_count, non_sampled_count


def _camera_passthrough_dict(action):
    if action and "engine_passthrough_blocks" in action:
        try:
            return json.loads(str(action["engine_passthrough_blocks"]))
        except Exception:
            return {}
    return {}


def _block_from_passthrough(pt_dict, block_name):
    try:
        blob = pt_dict.get(str(BLOCK_HASHES[block_name]))
        return base64.b64decode(blob) if blob else None
    except Exception:
        return None


def _stored_resident_timing(targets):
    for target in targets:
        if not target:
            continue
        try:
            count = int(target.get("engine_clip_resident_sample_count", 0))
            fps = float(target.get("engine_clip_resident_fps", 0.0))
        except Exception:
            continue
        if count > 0 and fps > 0.0:
            return count, fps
    return 0, 0.0

def _camera_import_timing_matches(targets, sample_count, fps):
    for target in targets:
        if not target:
            continue
        try:
            imported_count = int(target.get("engine_clip_import_sample_count", 0))
            imported_fps = float(target.get("engine_clip_import_fps", 0.0))
        except Exception:
            continue
        if imported_count > 0 and imported_fps > 0.0:
            return int(sample_count) == imported_count and abs(float(fps) - imported_fps) <= 0.01
    return False


def _camera_export_clear_flag_bits():
    # Camera clips must shed every "other type" flag and every skeletal-only
    # bit. Anything left here would be junk that survived from an imported
    # original (or from a clip the user reassigned to a camera).
    return (
        FLAG_CURVES
        | FLAG_FACIAL_POSES
        | FLAG_PERFORMANCE
        | FLAG_LOCATOR
        | FLAG_USER_POSE
        | FLAG_FRAME_DATA_LOOKUP
        | FLAG_STREAM_FRAME_DATA
        | FLAG_UNCOMPRESSED
        | FLAG_HAS_ANIM_MORPH
        | FLAG_HAS_ANIM_GEOM
        | FLAG_HAS_FACIAL
        | FLAG_HAS_ANIM_ZIVA
        | FLAG_HAS_MOTION_SAMPLES
        | FLAG_HAS_GEOM_CACHE
        | FLAG_MOTION_DELTA
        | FLAG_CONSTANT_PHASE
        | FLAG_HAS_PHASE
    )


def _sanitize_camera_export_flags(flags):
    # Force STD+CAMERA on, strip everything that doesn't belong in a camera
    # clip. Passed to reconcile_export_blocks so the AnimClipBuilt patch keeps
    # FLAG_CAMERA instead of being scrubbed by the skeletal sanitizer.
    return ((int(flags) & U32_MASK) | FLAG_STD | FLAG_CAMERA) & ~_camera_export_clear_flag_bits() & U32_MASK


def _camera_export_flags(original_flags, scene, trigger_count, has_custom_tracks, use_export_toggles=True):
    base = (int(original_flags) & U32_MASK) if original_flags is not None else (FLAG_STD | FLAG_CAMERA)
    if use_export_toggles:
        toggles = {
            "looping": bool(getattr(scene, "engine_export_looping", False)),
            "additive": bool(getattr(scene, "engine_export_additive", False)),
            "partial": bool(getattr(scene, "engine_export_partial", False)),
            "partial_motion": bool(getattr(scene, "engine_export_partial_motion", False)),
        }
        base = apply_flag_toggles(base, toggles)
    flags = _sanitize_camera_export_flags(base | FLAG_HAS_MOTION)
    if trigger_count > 0:
        flags |= FLAG_HAS_EVENTS
    else:
        flags &= ~FLAG_HAS_EVENTS
    if has_custom_tracks:
        flags |= FLAG_HAS_CUSTOM_TRACKS
    else:
        flags &= ~FLAG_HAS_CUSTOM_TRACKS
    return flags & U32_MASK


def _write_dat1(filepath, block_payload_by_hash, string_table=b"", add_stg_header=True):
    blocks = sorted(block_payload_by_hash.items(), key=lambda b: b[0])
    block_count = len(blocks)
    fixup_count = 0
    dat1_header_size = (
        DAT1_HEADER_SIZE
        + block_count * DAT1_BLOCK_TABLE_ENTRY_SIZE
        + fixup_count * DAT1_FIXUP_TABLE_ENTRY_SIZE
    )

    block_offsets = []
    cursor = dat1_header_size + len(string_table)
    block_payload_parts = [string_table] if string_table else []
    for _block_hash, payload in blocks:
        aligned = (cursor + DAT1_BLOCK_ALIGN - 1) & ~(DAT1_BLOCK_ALIGN - 1)
        pad = aligned - cursor
        if pad:
            block_payload_parts.append(b"\x00" * pad)
            cursor = aligned
        block_offsets.append(cursor)
        block_payload_parts.append(payload)
        cursor += len(payload)

    block_table = b"".join(
        struct.pack("<III", h, off, len(p)) for (h, p), off in zip(blocks, block_offsets)
    )
    body = b"".join(block_payload_parts)
    chunk_size = DAT1_HEADER_SIZE + len(block_table) + len(body)
    dat1_header = struct.pack(
        "<IIIHH",
        DAT1_FILE_ID,
        DAT1_VERSION_HASH_I30,
        chunk_size,
        block_count,
        fixup_count,
    )
    out = dat1_header + block_table + body

    asset_header = struct.pack("<IBBHII", DAT1_VERSION_HASH_I30, 0, 1, 0, chunk_size, 0)
    stg_buf = bytearray()
    stg_buf += struct.pack("<IIII", STG_MAGIC, STG_VERSION, len(asset_header), 0)
    stg_buf += asset_header
    rem = len(stg_buf) % STG_HEADER_ALIGN
    if rem != 0:
        stg_buf += b"\x00" * (STG_HEADER_ALIGN - rem)
    final_out = (bytes(stg_buf) + out) if add_stg_header else out
    with open(filepath, "wb") as f:
        f.write(final_out)
    return len(final_out), ("STG+DAT1" if add_stg_header else "raw DAT1")


def export_camera_animclip(context, operator, camera, filepath):
    if "sync_luna_settings_for_context" in globals():
        sync_luna_settings_for_context(context, force=False)
    scene = context.scene
    original_frame = scene.frame_current
    if getattr(scene, "use_smooth_playback", False):
        scene.use_smooth_playback = False
        scene.render.frame_map_old = 100
        scene.render.frame_map_new = 100

    frame_start = int(getattr(scene, "engine_export_frame_start", scene.frame_start))
    frame_end = int(getattr(scene, "engine_export_frame_end", scene.frame_end))
    if frame_end < frame_start:
        frame_end = frame_start
    timeline_sample_count = max(1, frame_end - frame_start + 1)
    fps = float(getattr(scene, "engine_anim_fps", scene.get("engine_base_fps", 0.0)))
    if fps <= 0.0:
        fps = scene.render.fps / max(1e-6, scene.render.fps_base)
    if fps <= 0.0:
        operator.report({"ERROR"}, "Export FPS must be greater than zero.")
        scene.frame_set(original_frame)
        return {"CANCELLED"}

    action = camera.animation_data.action if getattr(camera, "animation_data", None) else None
    if "_validate_clip_binding" in globals():
        cam_data = getattr(camera, "data", None)
        cam_data_action = cam_data.animation_data.action if cam_data and getattr(cam_data, "animation_data", None) else None
        _binding_id, binding_error = _validate_clip_binding(camera, action, cam_data, cam_data_action, label="Camera export target")
        if binding_error:
            operator.report({"ERROR"}, binding_error)
            scene.frame_set(original_frame)
            return {"CANCELLED"}
    use_original_values = bool(getattr(scene, "engine_export_use_original_values", True))
    migrate_clip_flags(camera, action)
    original_flags = get_original_flags(action, camera)
    if original_flags is None:
        original_flags = FLAG_STD | FLAG_CAMERA

    clip_sample_count, clip_fps, resident_sample_count, resident_fps = _resolve_export_timing(
        camera, action, timeline_sample_count, fps, original_flags
    )
    if use_original_values:
        stored_resident_count, stored_resident_fps = _stored_resident_timing((camera, action))
        if (
            stored_resident_count > 0
            and stored_resident_fps > 0.0
            and _camera_import_timing_matches((camera, action), timeline_sample_count, fps)
        ):
            resident_sample_count = stored_resident_count
            resident_fps = stored_resident_fps
    clip_sample_count = max(1, int(clip_sample_count))
    want_looping = bool(int(original_flags) & FLAG_LOOPING)
    if not use_original_values:
        want_looping = bool(getattr(scene, "engine_export_looping", False))

    sample_frames = _motion_sample_frames(frame_start, frame_end, clip_sample_count)
    samples_bl = _sample_camera_world(context, camera, sample_frames)

    pt_dict = _camera_passthrough_dict(action)
    template_motion = _block_from_passthrough(pt_dict, "AnimClipMotionData") or _motion_original_blob(action)
    motion_payload = _build_camera_motion_block(samples_bl, template_motion, is_looping=want_looping)

    clip_name = os.path.splitext(os.path.basename(filepath))[0]
    stored_name_hash = camera.get("engine_clip_name_hash")
    if use_original_values and stored_name_hash is not None:
        clip_name_hash = int(stored_name_hash) & U32_MASK
    else:
        clip_name_hash = string_crc32(clip_name)

    string_names = [CAMERA_TRACK_XFOV, CAMERA_TRACK_FOCAL_LENGTH, clip_name]
    specs = _camera_custom_track_specs(action)
    for spec in specs:
        name = str(spec.get("name") or "")
        if name and name not in string_names:
            string_names.append(name)
    string_table, string_offsets = _build_string_table(string_names)

    custom_tracks_payload, custom_track_data_payload, sampled_curve_count, non_sampled_curve_count = _build_camera_custom_track_blocks(
        context, camera, action, sample_frames, string_offsets
    )

    trigger_count = len(_action_trigger_marker_entries(action)) if action else 0
    clip_flags = _camera_export_flags(
        original_flags,
        scene,
        trigger_count,
        bool(custom_tracks_payload and custom_track_data_payload),
        use_export_toggles=not use_original_values,
    )
    duration = _duration_from_sample_timing(clip_sample_count, clip_fps, bool(clip_flags & FLAG_LOOPING))
    cycle_count = int(camera.get("engine_clip_cycle_count", 1)) & 0xFF if use_original_values else 1
    facial_name_hash = int(camera.get("engine_clip_facial_name_hash", 0)) & U32_MASK if use_original_values else 0
    sound_event_hash = int(camera.get("engine_clip_sound_event_hash", DEFAULT_SOUND_EVENT_HASH)) & U32_MASK if use_original_values else DEFAULT_SOUND_EVENT_HASH
    particle_count_max = int(camera.get("engine_clip_particle_count_max", 0)) & U32_MASK if use_original_values else 0

    anim_clip_built = struct.pack(
        ANIM_CLIP_BUILT_FORMAT,
        int(string_offsets.get(clip_name, 0)) & U32_MASK,
        clip_name_hash,
        clip_flags,
        float(clip_fps),
        float(duration),
        cycle_count,
        0, 0, 0,
        0,
        0,
        0,
        clip_sample_count,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        int(sampled_curve_count) & 0xFFFF,
        int(non_sampled_curve_count) & 0xFFFF,
        max(1, int(resident_sample_count)),
        float(resident_fps if resident_fps > 0.0 else clip_fps),
        0,
        facial_name_hash,
        sound_event_hash,
        particle_count_max,
        b"\x00" * ANIM_CLIP_BUILT_RESERVED_BYTES,
    )

    out_blocks = {
        BLOCK_HASHES["AnimClipBuilt"]: anim_clip_built,
        BLOCK_HASHES["AnimClipMotionData"]: motion_payload,
    }
    if custom_tracks_payload and custom_track_data_payload:
        out_blocks[BLOCK_HASHES["AnimClipCustomTracks"]] = custom_tracks_payload
        out_blocks[BLOCK_HASHES["AnimClipCustomTrackData"]] = custom_track_data_payload

    skip_hashes = {
        BLOCK_HASHES["AnimClipBuilt"],
        BLOCK_HASHES["AnimClipMotionData"],
        BLOCK_HASHES["AnimClipCustomTracks"],
        BLOCK_HASHES["AnimClipCustomTrackData"],
    }
    for h_str, b64_data in pt_dict.items():
        try:
            block_hash = int(h_str)
        except Exception:
            continue
        if block_hash in skip_hashes:
            continue
        try:
            out_blocks[block_hash] = base64.b64decode(b64_data)
        except Exception:
            pass

    clip_flags, export_warnings, export_errors = reconcile_export_blocks(
        out_blocks, clip_flags, sanitize_fn=_sanitize_camera_export_flags,
    )
    for warning in export_warnings:
        operator.report({"WARNING"}, warning)
    if export_errors:
        operator.report({"ERROR"}, " ".join(export_errors))
        scene.frame_set(original_frame)
        return {"CANCELLED"}

    set_current_flags(camera, clip_flags)
    if action:
        set_current_flags(action, clip_flags)

    add_stg_header = bool(getattr(scene, "engine_export_add_stg_header", True))
    try:
        _size, format_name = _write_dat1(filepath, out_blocks, string_table=string_table, add_stg_header=add_stg_header)
    except OSError as exc:
        operator.report({"ERROR"}, f"Write failed: {exc}")
        scene.frame_set(original_frame)
        return {"CANCELLED"}

    scene.frame_set(original_frame)
    context.window_manager.progress_update(100)
    operator.report({"INFO"}, f"Wrote camera {format_name}: {clip_sample_count} samples, {sampled_curve_count + non_sampled_curve_count} custom tracks")
    return {"FINISHED"}
