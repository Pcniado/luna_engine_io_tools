# This module is part of the public-release split of blender_import_model_anim_release.py.
# Behavior-sensitive logic was moved mechanically; avoid algorithm changes here.

from .utils import *

MOTION_EMPTY_BLOCK_SIZE = 0x80
MOTION_TYPE_OFFSET = 0x00
MOTION_INITIAL_ROT0_OFFSET = 0x04
MOTION_INITIAL_ROT1_OFFSET = 0x14
MOTION_INITIAL_TRANS0_OFFSET = 0x24
MOTION_INITIAL_TRANS1_OFFSET = 0x30
MOTION_STREAM_HEADER_OFFSET = 0x3C
MOTION_BOUNDS_MIN_OFFSET = 0x4C
MOTION_BOUNDS_MAX_OFFSET = 0x64
MOTION_EMPTY_SEGMENT_OFFSET = 4
MOTION_HEADER_PAD_BYTES = 5
TRACKS_DATA_PAD_BYTES = 22
STANDARD_SAMPLE_QUAT_DECODE_SCALE = 32768.0
STANDARD_SAMPLE_QUAT_ENCODE_SCALE = 32767.0
STANDARD_SAMPLE_LOG_SCALE_MIN = 0
STANDARD_SAMPLE_LOG_SCALE_MAX = 16
STANDARD_SAMPLE_PHASE_MIN = -128
STANDARD_SAMPLE_PHASE_MAX = 127
STANDARD_SAMPLE_I16_MIN = -32768
STANDARD_SAMPLE_I16_MAX = 32767


def _motion_identity_block():
    # Standard AnimMotionData header is 0x7C bytes. The empty sample stream pads
    # to 0x80, and SegmentDataOffset is relative to m_Data at 0x7C.
    data = bytearray(MOTION_EMPTY_BLOCK_SIZE)
    struct.pack_into("<BBH", data, MOTION_TYPE_OFFSET, ANIM_MOTION_TYPE_STANDARD, 0, 0)
    struct.pack_into("<ffff", data, MOTION_INITIAL_ROT0_OFFSET, 0.0, 0.0, 0.0, 1.0)
    struct.pack_into("<ffff", data, MOTION_INITIAL_ROT1_OFFSET, 0.0, 0.0, 0.0, 1.0)
    struct.pack_into("<fff", data, MOTION_INITIAL_TRANS0_OFFSET, 0.0, 0.0, 0.0)
    struct.pack_into("<fff", data, MOTION_INITIAL_TRANS1_OFFSET, 0.0, 0.0, 0.0)
    struct.pack_into(
        "<IIHB5s",
        data,
        MOTION_STREAM_HEADER_OFFSET,
        0,
        MOTION_EMPTY_SEGMENT_OFFSET,
        0,
        0,
        b"\x00" * MOTION_HEADER_PAD_BYTES,
    )
    return bytes(data)

def _empty_tracks_data_block(locator_count=0, trigger_count=0, event_data_size=0, marker_count=0):
    return struct.pack(
        ANIM_CLIP_TRACKS_DATA_FORMAT,
        int(locator_count) & 0xFFFF,
        int(trigger_count) & 0xFFFF,
        int(event_data_size) & U32_MASK,
        int(marker_count) & 0xFFFF,
        b"\x00" * TRACKS_DATA_PAD_BYTES,
    )

def _coerce_tracks_data_block(block):
    data = bytearray(block or b"")
    if len(data) < ANIM_CLIP_TRACKS_DATA_SIZE:
        data.extend(b"\x00" * (ANIM_CLIP_TRACKS_DATA_SIZE - len(data)))
    return data

def _read_tracks_counts(block):
    data = _coerce_tracks_data_block(block)
    locator_count, trigger_count, event_data_size, marker_count, _pad = struct.unpack_from(ANIM_CLIP_TRACKS_DATA_FORMAT, data, 0)
    return locator_count, trigger_count, event_data_size, marker_count, data

def _read_motion_header(block):
    if not block or len(block) < ANIM_MOTION_HEADER_SIZE:
        return {"present": bool(block), "supported": False, "reason": "too small"}
    try:
        mtype, flags, sample_count = struct.unpack_from("<BBH", block, MOTION_TYPE_OFFSET)
        initial_rot0 = struct.unpack_from("<ffff", block, MOTION_INITIAL_ROT0_OFFSET)
        initial_rot1 = struct.unpack_from("<ffff", block, MOTION_INITIAL_ROT1_OFFSET)
        initial_trans0 = struct.unpack_from("<fff", block, MOTION_INITIAL_TRANS0_OFFSET)
        initial_trans1 = struct.unpack_from("<fff", block, MOTION_INITIAL_TRANS1_OFFSET)
        sample_off, segment_off, phase_blend, segment_count, _pad = struct.unpack_from(
            "<IIHB5s", block, MOTION_STREAM_HEADER_OFFSET
        )
    except struct.error:
        return {"present": True, "supported": False, "reason": "truncated header"}

    sample_size = ANIM_MOTION_STANDARD_SAMPLE_SIZE if mtype == ANIM_MOTION_TYPE_STANDARD else ANIM_MOTION_FULL_SAMPLE_SIZE
    sample_start = ANIM_MOTION_HEADER_SIZE + sample_off
    sample_end = sample_start + (sample_count * sample_size)
    segment_start = ANIM_MOTION_HEADER_SIZE + segment_off
    supported = (
        mtype == ANIM_MOTION_TYPE_STANDARD
        and (flags & ANIM_MOTION_FLAG_HAS_SECONDARY) == 0
        and sample_start >= ANIM_MOTION_HEADER_SIZE
        and sample_end <= len(block)
        and segment_start <= len(block)
    )
    reason = ""
    if mtype != ANIM_MOTION_TYPE_STANDARD:
        reason = "full motion samples are preserved but not editable"
    elif flags & ANIM_MOTION_FLAG_HAS_SECONDARY:
        reason = "secondary motion stream is preserved but not editable"
    elif sample_end > len(block):
        reason = "sample stream is outside the block"
    elif segment_start > len(block):
        reason = "segment stream is outside the block"

    return {
        "present": True,
        "supported": supported,
        "reason": reason,
        "type": mtype,
        "flags": flags,
        "sample_count": sample_count,
        "sample_offset": sample_off,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "segment_offset": segment_off,
        "segment_start": segment_start,
        "phase_blend_samples": phase_blend,
        "segment_count": segment_count,
        "initial_rot0": initial_rot0,
        "initial_rot1": initial_rot1,
        "initial_trans0": initial_trans0,
        "initial_trans1": initial_trans1,
    }

def _decode_standard_motion_sample(block, offset):
    qx, qy, qz, qw, tx, ty, tz, log_scale, phase = struct.unpack_from("<hhhhhhhbb", block, offset)
    quat = mathutils.Quaternion((
        qw / STANDARD_SAMPLE_QUAT_DECODE_SCALE,
        qx / STANDARD_SAMPLE_QUAT_DECODE_SCALE,
        qy / STANDARD_SAMPLE_QUAT_DECODE_SCALE,
        qz / STANDARD_SAMPLE_QUAT_DECODE_SCALE,
    ))
    quat.normalize()
    scale = float(1 << max(STANDARD_SAMPLE_LOG_SCALE_MIN, min(STANDARD_SAMPLE_LOG_SCALE_MAX, int(log_scale))))
    loc = mathutils.Vector((tx / scale, ty / scale, tz / scale))
    return loc, quat, phase

def _pack_standard_motion_sample(loc, quat, phase=0):
    quat = quat.copy()
    quat.normalize()
    max_abs = max(abs(loc.x), abs(loc.y), abs(loc.z))
    log_scale = _compute_trans_log_scale(max_abs)
    scale = float(1 << log_scale)

    def clamp_i16(value):
        return max(STANDARD_SAMPLE_I16_MIN, min(STANDARD_SAMPLE_I16_MAX, int(round(value))))

    return struct.pack(
        "<hhhhhhhbb",
        clamp_i16(quat.x * STANDARD_SAMPLE_QUAT_ENCODE_SCALE),
        clamp_i16(quat.y * STANDARD_SAMPLE_QUAT_ENCODE_SCALE),
        clamp_i16(quat.z * STANDARD_SAMPLE_QUAT_ENCODE_SCALE),
        clamp_i16(quat.w * STANDARD_SAMPLE_QUAT_ENCODE_SCALE),
        clamp_i16(loc.x * scale),
        clamp_i16(loc.y * scale),
        clamp_i16(loc.z * scale),
        max(STANDARD_SAMPLE_LOG_SCALE_MIN, min(STANDARD_SAMPLE_LOG_SCALE_MAX, int(log_scale))),
        max(STANDARD_SAMPLE_PHASE_MIN, min(STANDARD_SAMPLE_PHASE_MAX, int(phase))),
    )

def engine_motion_to_blender_transform(loc, quat):
    swz = SWIZZLE_MAT.to_quaternion()
    swz_inv = swz.inverted()
    bl_loc = SWIZZLE_MAT @ loc
    bl_quat = swz @ quat @ swz_inv
    bl_quat.normalize()
    return bl_loc, bl_quat

def blender_transform_to_engine_motion(loc, quat):
    swz = SWIZZLE_MAT.to_quaternion()
    swz_inv = swz.inverted()
    eng_loc = SWIZZLE_MAT.inverted() @ loc
    eng_quat = swz_inv @ quat @ swz
    eng_quat.normalize()
    return eng_loc, eng_quat

def _matrix_from_loc_quat(loc, quat):
    return mathutils.Matrix.Translation(loc) @ quat.to_matrix().to_4x4()

def _loc_quat_from_matrix(matrix):
    loc, quat, _scale = matrix.decompose()
    quat.normalize()
    return loc, quat

def _clear_fcurve_points(fc):
    try:
        fc.keyframe_points.clear()
        return
    except Exception:
        pass
    try:
        while fc.keyframe_points:
            fc.keyframe_points.remove(fc.keyframe_points[0], fast=True)
    except Exception:
        pass

def _ensure_action_fcurve(action, datablock, data_path, index):
    if hasattr(action, "fcurve_ensure_for_datablock"):
        fc = action.fcurve_ensure_for_datablock(
            datablock=datablock, data_path=data_path, index=index
        )
        _clear_fcurve_points(fc)
        return fc

    for fc in getattr(action, "fcurves", []):
        if fc.data_path == data_path and fc.array_index == index:
            _clear_fcurve_points(fc)
            return fc
    return action.fcurves.new(data_path=data_path, index=index)

def _find_action_fcurve(action, datablock, data_path, index):
    if hasattr(action, "fcurves"):
        return action.fcurves.find(data_path, index=index)
    if datablock and hasattr(action, "fcurve_find_for_datablock"):
        try:
            return action.fcurve_find_for_datablock(
                datablock=datablock, data_path=data_path, index=index
            )
        except Exception:
            return None
    return None

def _root_motion_empty_poll(_self, obj):
    return obj is not None and obj.type == 'EMPTY'

def _unique_object_name(base_name):
    if base_name not in bpy.data.objects:
        return base_name
    idx = 1
    while f"{base_name}.{idx:03d}" in bpy.data.objects:
        idx += 1
    return f"{base_name}.{idx:03d}"

def _try_set_idprop(datablock, key, value):

    if datablock is None:
        return False
    try:
        datablock[key] = value
        return True
    except (AttributeError, RuntimeError, TypeError):
        return False

def _idprop_string(datablock, key, default=""):
    if datablock is None:
        return default
    try:
        value = datablock.get(key, default)
    except Exception:
        return default
    if value is None:
        return default
    return str(value)

def _new_clip_binding_id():
    return uuid.uuid4().hex

def _clip_binding_id(*targets):
    for target in targets:
        value = _idprop_string(target, "engine_clip_binding_id", "")
        if value:
            return value
    return ""

def _ensure_clip_binding_id(*targets, binding_id=None):
    binding_id = str(binding_id or "") or _clip_binding_id(*targets) or _new_clip_binding_id()
    for target in targets:
        if target is not None:
            _try_set_idprop(target, "engine_clip_binding_id", binding_id)
    return binding_id

def _binding_target_label(target):
    if target is None:
        return "<none>"
    name = getattr(target, "name", "") or "<unnamed>"
    target_type = getattr(target, "type", None) or type(target).__name__
    return f"{target_type} {name}"

def _validate_clip_binding(*targets, label="AnimClip"):
    present = []
    missing = []
    for target in targets:
        if target is None:
            continue
        value = _clip_binding_id(target)
        if value:
            present.append((target, value))
        else:
            missing.append(target)
    if missing:
        names = ", ".join(_binding_target_label(t) for t in missing)
        return "", f"{label} is missing Luna binding metadata on: {names}. Re-import with the updated add-on so export cannot borrow another actor's motion."
    values = {value for _target, value in present}
    if len(values) > 1:
        names = ", ".join(f"{_binding_target_label(t)}={v}" for t, v in present)
        return "", f"{label} has mismatched Luna binding metadata: {names}."
    return (present[0][1] if present else ""), ""

def _root_motion_empty_binding_id(empty):
    return _idprop_string(empty, "engine_root_motion_binding_id", "") or _clip_binding_id(empty)

def _root_motion_empty_matches(empty, binding_id):
    return bool(
        empty is not None
        and getattr(empty, "type", None) == 'EMPTY'
        and binding_id
        and _root_motion_empty_binding_id(empty) == binding_id
    )

def _find_bound_root_motion_empty(binding_id):
    if not binding_id:
        return None, "Root-motion binding id is missing."
    matches = []
    for obj in bpy.data.objects:
        if _root_motion_empty_matches(obj, binding_id):
            matches.append(obj)
    if len(matches) == 1:
        return matches[0], ""
    if len(matches) > 1:
        names = ", ".join(obj.name for obj in matches)
        return None, f"Root-motion binding {binding_id} is ambiguous; matching Empties: {names}."
    return None, f"No root-motion Empty is bound to Luna binding {binding_id}."

def _resolve_bound_root_motion_empty(armature=None, action=None):
    binding_id, error = _validate_clip_binding(armature, action, label="Root-motion owner")
    if error:
        return None, error

    for owner in (armature, action):
        name = _idprop_string(owner, "engine_root_motion_empty_name", "")
        if not name:
            continue
        obj = bpy.data.objects.get(name)
        if not obj:
            return None, f"Bound root-motion Empty '{name}' no longer exists for {_binding_target_label(owner)}."
        if getattr(obj, "type", None) != 'EMPTY':
            return None, f"Bound root-motion object '{name}' is not an Empty."
        if not _root_motion_empty_matches(obj, binding_id):
            return None, (
                f"Root-motion Empty '{name}' is bound to '{_root_motion_empty_binding_id(obj) or '<missing>'}', "
                f"but {_binding_target_label(owner)} needs '{binding_id}'."
            )
        return obj, ""

    return _find_bound_root_motion_empty(binding_id)

def _bind_root_motion_empty(scene, armature, empty, action=None, update_scene_pointer=False):
    if not empty or empty.type != 'EMPTY':
        return None

    binding_id = _ensure_clip_binding_id(armature, action)
    _ensure_clip_binding_id(empty, binding_id=binding_id)
    if action:
        _try_set_idprop(action, "engine_root_motion_empty_name", empty.name)
    _try_set_idprop(empty, "engine_root_motion_controller", True)
    _try_set_idprop(empty, "engine_root_motion_binding_id", binding_id)
    if armature:
        _try_set_idprop(empty, "engine_root_motion_armature_name", armature.name)
        _try_set_idprop(armature, "engine_root_motion_empty_name", empty.name)

    if update_scene_pointer and scene and hasattr(scene, "engine_root_motion_empty"):
        try:
            scene.engine_root_motion_empty = empty
        except Exception:
            pass
    return empty

def _resolve_root_motion_empty(scene=None, armature=None, action=None):
    if not armature and scene and hasattr(scene, "engine_root_motion_empty"):
        try:
            obj = scene.engine_root_motion_empty
            if obj and obj.type == 'EMPTY' and _root_motion_empty_binding_id(obj):
                return obj
        except Exception:
            pass
        return None

    empty, _error = _resolve_bound_root_motion_empty(armature, action)
    return empty

def _ensure_root_motion_empty(scene, armature=None, action=None, create=False, parent_armature=False):
    if armature or action:
        _ensure_clip_binding_id(armature, action)
    empty = _resolve_root_motion_empty(scene, armature, action)
    created = False
    if not empty and armature is not None:
        parent = getattr(armature, "parent", None)
        if parent and getattr(parent, "type", None) == 'EMPTY':
            empty = parent
    if not empty and create:
        collection = None
        if armature and getattr(armature, "users_collection", None):
            collection = armature.users_collection[0]
        if collection is None:
            collection = bpy.context.collection
        base_name = f"{ROOT_MOTION_EMPTY_BASENAME}_{armature.name}" if armature else ROOT_MOTION_EMPTY_BASENAME
        empty = bpy.data.objects.new(_unique_object_name(base_name), None)
        empty.empty_display_type = 'ARROWS'
        empty.empty_display_size = 1.0
        empty.show_name = True
        collection.objects.link(empty)
        empty.matrix_world = mathutils.Matrix.Identity(4)
        created = True
    if not empty:
        return None

    update_scene_pointer = created
    if scene and hasattr(scene, "engine_root_motion_empty"):
        try:
            if scene.engine_root_motion_empty is None:
                update_scene_pointer = True
        except Exception:
            pass
    _bind_root_motion_empty(scene, armature, empty, action=action, update_scene_pointer=update_scene_pointer)
    if action:
        _try_set_idprop(empty, "engine_root_motion_source_action", action.name)
    if parent_armature and armature and armature.parent != empty:
        arm_world = armature.matrix_world.copy()
        armature.parent = empty
        armature.matrix_world = arm_world
    return empty

def _ensure_root_motion_action(empty, base_action=None):
    if not empty:
        return None
    binding_id = _ensure_clip_binding_id(base_action, empty)
    if not empty.animation_data:
        empty.animation_data_create()
    if empty.animation_data.action and empty.animation_data.action != base_action:
        _ensure_clip_binding_id(empty.animation_data.action, binding_id=binding_id)
        return empty.animation_data.action
    base_name = base_action.name if base_action else empty.name
    action = bpy.data.actions.new(name=f"{base_name}_root_motion")
    action["engine_root_motion_action"] = True
    _ensure_clip_binding_id(action, binding_id=binding_id)
    if base_action:
        action["engine_root_motion_source_action"] = base_action.name
    empty.animation_data.action = action
    return action

def _transform_action_for_object(obj):
    if not obj or not obj.animation_data:
        return None
    return obj.animation_data.action

def _root_motion_source_object(armature, scene=None, action=None, allow_legacy=False):
    empty = _resolve_root_motion_empty(scene or bpy.context.scene, armature, action)
    if empty:
        return empty
    return armature if allow_legacy else None

def _remove_transform_fcurves(action):
    if not action or not hasattr(action, "fcurves"):
        return
    paths = {"location", "rotation_quaternion", "rotation_euler"}
    for fc in list(action.fcurves):
        if fc.data_path in paths:
            try:
                action.fcurves.remove(fc)
            except Exception:
                pass

def _write_root_motion_samples_to_object(empty, base_action, frames, samples):
    if not empty or not samples:
        return None
    action = _ensure_root_motion_action(empty, base_action)
    if not action:
        return None
    empty.rotation_mode = 'QUATERNION'
    _remove_transform_fcurves(action)

    count = len(samples)
    loc_values = [[] for _ in range(3)]
    rot_values = [[] for _ in range(4)]
    for loc, quat in samples:
        loc_values[0].append(loc.x)
        loc_values[1].append(loc.y)
        loc_values[2].append(loc.z)
        rot_values[0].append(quat.w)
        rot_values[1].append(quat.x)
        rot_values[2].append(quat.y)
        rot_values[3].append(quat.z)

    for idx in range(3):
        fc = _ensure_action_fcurve(action, empty, "location", idx)
        fc.keyframe_points.add(count)
        co = np.empty(count * 2, dtype=np.float32)
        co[0::2] = frames
        co[1::2] = loc_values[idx]
        fc.keyframe_points.foreach_set("co", co)
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'
        fc.update()

    for idx in range(4):
        fc = _ensure_action_fcurve(action, empty, "rotation_quaternion", idx)
        fc.keyframe_points.add(count)
        co = np.empty(count * 2, dtype=np.float32)
        co[0::2] = frames
        co[1::2] = rot_values[idx]
        fc.keyframe_points.foreach_set("co", co)
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'
        fc.update()

    empty.location = samples[0][0]
    empty.rotation_quaternion = samples[0][1]
    return action

def object_motion_fcurves(action, datablock=None):
    if not action:
        return []
    paths = {"location", "rotation_quaternion", "rotation_euler"}
    if hasattr(action, "fcurves"):
        return [fc for fc in action.fcurves if fc.data_path in paths]


    out = []
    for data_path, count in (("location", 3), ("rotation_quaternion", 4), ("rotation_euler", 3)):
        for index in range(count):
            fc = _find_action_fcurve(action, datablock, data_path, index)
            if fc:
                out.append(fc)
    return out

def object_motion_has_keys(armature):
    source = _root_motion_source_object(armature, action=_resolve_anim_action(armature))
    action = _transform_action_for_object(source)
    return any(len(fc.keyframe_points) > 0 for fc in object_motion_fcurves(action, source))

def _iter_obj_action_fcurves(obj, action):
    if not action:
        return
    anim_data = getattr(obj, "animation_data", None)
    bound_slot = getattr(anim_data, "action_slot", None) if anim_data else None
    bound_handle = getattr(bound_slot, "handle", None) if bound_slot else None

    layers = getattr(action, "layers", None)
    if layers is not None:
        # Try slot-matched channelbag first
        for layer in layers:
            for strip in getattr(layer, "strips", []) or []:
                channelbags = getattr(strip, "channelbags", None)
                if channelbags is None:
                    continue
                for cb in channelbags:
                    slot = getattr(cb, "slot", None) or getattr(cb, "action_slot", None)
                    slot_handle = getattr(slot, "handle", None) if slot else None
                    is_match = False
                    if bound_handle is not None and slot_handle is not None:
                        is_match = (slot_handle == bound_handle)
                    elif slot is not None:
                        slot_name = (getattr(slot, "name_display", None)
                                     or getattr(slot, "name", None) or "")
                        is_match = (str(slot_name) == obj.name)
                    if is_match:
                        for fc in getattr(cb, "fcurves", []) or []:
                            yield fc
                        return  # Only iterate the matching channelbag

        # No slot-bound match found; fall back to iterating every channelbag.
        for layer in layers:
            for strip in getattr(layer, "strips", []) or []:
                for cb in getattr(strip, "channelbags", []) or []:
                    for fc in getattr(cb, "fcurves", []) or []:
                        yield fc
        return

    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        for fc in legacy:
            yield fc

def _find_obj_fcurve(obj, action, data_path, index):
    if not action:
        return None
    for fc in _iter_obj_action_fcurves(obj, action):
        if fc.data_path == data_path and fc.array_index == index:
            return fc
    return None

def _motion_sample_frames(frame_start, frame_end, sample_count):
    count = max(1, int(sample_count))
    start = float(frame_start)
    end = float(frame_end)
    if count == 1:
        return [start]
    if abs(end - start) <= 1e-6:
        return [start + float(i) for i in range(count)]
    step = (end - start) / float(count - 1)
    return [start + (step * float(i)) for i in range(count)]

def _sample_object_transform(obj, frame):
    action = obj.animation_data.action if (obj and obj.animation_data) else None
    loc = obj.location.copy()
    rot_mode = obj.rotation_mode
    if rot_mode == 'QUATERNION':
        quat = obj.rotation_quaternion.copy()
    else:
        quat = obj.rotation_euler.to_quaternion()
    if not action:
        quat.normalize()
        return loc, quat

    for axis in range(3):
        fc = _find_obj_fcurve(obj, action, "location", axis)
        if fc:
            loc[axis] = fc.evaluate(float(frame))

    new_quat = [quat.w, quat.x, quat.y, quat.z]
    quat_found = False
    for axis in range(4):
        fc = _find_obj_fcurve(obj, action, "rotation_quaternion", axis)
        if fc:
            new_quat[axis] = fc.evaluate(float(frame))
            quat_found = True
    if quat_found:
        quat = mathutils.Quaternion((new_quat[0], new_quat[1], new_quat[2], new_quat[3]))
    else:
        euler = obj.rotation_euler.copy()
        eul_found = False
        for axis in range(3):
            fc = _find_obj_fcurve(obj, action, "rotation_euler", axis)
            if fc:
                euler[axis] = fc.evaluate(float(frame))
                eul_found = True
        if eul_found:
            quat = euler.to_quaternion()

    quat.normalize()
    return loc, quat

def _sample_object_transform_via_depsgraph(obj, frame):
    scene = bpy.context.scene
    frame_float = float(frame)
    frame_int = math.floor(frame_float)
    scene.frame_set(int(frame_int), subframe=frame_float - float(frame_int))
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
    except Exception:
        eval_obj = obj
    mw = eval_obj.matrix_world.copy()
    parent = getattr(obj, "parent", None)
    if parent:
        try:
            mw = parent.matrix_world.inverted() @ mw
        except Exception:
            pass
    loc, quat, _scale = mw.decompose()
    quat.normalize()
    return loc, quat

def sample_object_root_motion_at_frames(armature, frames):
    source = _root_motion_source_object(armature, action=_resolve_anim_action(armature))
    if not source:
        return []
    frame_values = [float(frame) for frame in frames]
    samples = [
        _sample_object_transform(source, frame)
        for frame in frame_values
    ]

    if samples and _root_motion_samples_are_flat(samples):
        original_frame = bpy.context.scene.frame_current
        original_subframe = getattr(bpy.context.scene, "frame_subframe", 0.0)
        try:
            samples = [
                _sample_object_transform_via_depsgraph(source, frame)
                for frame in frame_values
            ]
        finally:
            bpy.context.scene.frame_set(original_frame, subframe=original_subframe)
    return samples

def sample_object_root_motion(armature, frame_start, frame_end, fps=None):
    frames = [float(frame) for frame in range(int(frame_start), int(frame_end) + 1)]
    return sample_object_root_motion_at_frames(armature, frames)

def compute_object_motion_signature(armature, frame_start, frame_end):
    digest = hashlib.sha1()
    for loc, quat in sample_object_root_motion(armature, frame_start, frame_end):
        digest.update(struct.pack(
            "<7f",
            round(float(loc.x), 6), round(float(loc.y), 6), round(float(loc.z), 6),
            round(float(quat.w), 6), round(float(quat.x), 6),
            round(float(quat.y), 6), round(float(quat.z), 6),
        ))
    return digest.hexdigest()

def _root_motion_import_signature(armature=None, source=None, source_action=None, action=None):
    for target in (source_action, source, action, armature):
        if not target:
            continue
        try:
            value = str(target.get("engine_motion_import_signature", "") or "")
        except Exception:
            value = ""
        if value:
            return value
    return ""

def _root_motion_original_present(armature=None, source=None, source_action=None, action=None):
    for target in (action, source_action, source, armature):
        if not target:
            continue
        try:
            if bool(target.get("engine_motion_original_present")):
                return True
        except Exception:
            pass
    return False

def _root_motion_signature_from_samples(samples):
    digest = hashlib.sha1()
    for loc, quat in samples or []:
        digest.update(struct.pack(
            "<7f",
            round(float(loc.x), 6), round(float(loc.y), 6), round(float(loc.z), 6),
            round(float(quat.w), 6), round(float(quat.x), 6),
            round(float(quat.y), 6), round(float(quat.z), 6),
        ))
    return digest.hexdigest()

def object_motion_matches_imported(armature, frame_start, frame_end):
    action_owner = _resolve_anim_action(armature)
    source = _root_motion_source_object(armature, action=action_owner)
    action = _transform_action_for_object(source)
    expected = _root_motion_import_signature(armature, source, action, action_owner)
    return bool(expected and compute_object_motion_signature(armature, frame_start, frame_end) == expected)

def _root_motion_samples_are_flat(samples):
    if not samples:
        return True
    base_loc, base_quat = samples[0]
    for loc, quat in samples[1:]:
        if (loc - base_loc).length > 1e-5:
            return False
        delta = abs(float(base_quat.dot(quat)))
        if abs(1.0 - delta) > 1e-5:
            return False
    return True

def object_motion_is_flat(armature, frame_start, frame_end):
    return _root_motion_samples_are_flat(sample_object_root_motion(armature, frame_start, frame_end))

def _object_motion_status(armature, frame_start, frame_end):
    if not armature:
        return "Missing"

    owner_action = _resolve_anim_action(armature)
    source = _root_motion_source_object(armature, action=owner_action)
    source_action = _transform_action_for_object(source)
    imported_sig = _root_motion_import_signature(armature, source, source_action, owner_action)
    original_present = _root_motion_original_present(armature, source, source_action, owner_action)

    try:
        has_keys = object_motion_has_keys(armature)
    except Exception:
        has_keys = False

    try:
        samples = sample_object_root_motion(armature, frame_start, frame_end)
    except Exception:
        return "Unknown"

    if not samples:
        return "Missing" if not imported_sig else "Flat"

    is_flat = _root_motion_samples_are_flat(samples)
    if is_flat:
        if has_keys or imported_sig or original_present:
            return "Flat"
        return "Missing"

    if imported_sig:
        stored_count = 0
        stored_start = frame_start
        stored_end = frame_end
        for target in (source_action, source, _resolve_anim_action(armature), armature):
            if not target:
                continue
            try:
                if int(target.get("engine_motion_import_sample_count", 0)) > 0:
                    stored_count = int(target.get("engine_motion_import_sample_count", 0))
                    stored_start = float(target.get("engine_motion_import_frame_start", frame_start))
                    stored_end = float(target.get("engine_motion_import_frame_end", frame_end))
                    break
            except Exception:
                pass
        if stored_count > 0:
            stored_frames = _motion_sample_frames(stored_start, stored_end, stored_count)
            stored_sig = _root_motion_signature_from_samples(
                sample_object_root_motion_at_frames(armature, stored_frames)
            )
            if stored_sig == imported_sig:
                return "Unchanged"

    current_sig = _root_motion_signature_from_samples(samples)
    if imported_sig and current_sig == imported_sig:
        return "Unchanged"

    return "Edited"

def _object_motion_status_cached(arm, action):
    if not arm:
        return "Missing"
    try:
        has_keys = object_motion_has_keys(arm)
    except Exception:
        has_keys = False
    source = _root_motion_source_object(arm, action=action)
    source_action = _transform_action_for_object(source)
    imported_sig = str(source_action.get("engine_motion_import_signature", "")) if source_action else ""
    if not imported_sig:
        imported_sig = str(action.get("engine_motion_import_signature", "")) if action else ""
    if not imported_sig:
        imported_sig = str(arm.get("engine_motion_import_signature", "")) if arm else ""
    original_present = bool(
        (action and action.get("engine_motion_original_present")) or
        (source_action and source_action.get("engine_motion_original_present")) or
        (source and source.get("engine_motion_original_present")) or
        (arm and arm.get("engine_motion_original_present"))
    )
    if has_keys:
        return "Imported" if imported_sig else "Present"
    if imported_sig or original_present:
        return "Missing / Flat"
    return "Missing"

def _motion_original_blob(action):
    if action and "engine_motion_original_blob" in action:
        try:
            return base64.b64decode(str(action["engine_motion_original_blob"]))
        except Exception:
            pass
    if action and "engine_passthrough_blocks" in action:
        try:
            pt = json.loads(str(action["engine_passthrough_blocks"]))
            blob = pt.get(str(BLOCK_HASHES["AnimClipMotionData"]))
            if blob:
                return base64.b64decode(blob)
        except Exception:
            return None
    return None

def _store_motion_metadata(armature, action, block, supported, reason, sample_count=0, frame_start=0, frame_end=0, signature=""):
    blob = base64.b64encode(block or b"").decode("ascii") if block else ""
    targets = [t for t in (armature, action) if t is not None]
    if targets:
        _ensure_clip_binding_id(*targets)
    for target in targets:
        target["engine_motion_original_present"] = bool(block)
        target["engine_motion_supported"] = bool(supported)
        target["engine_motion_original_size"] = int(len(block or b""))
        target["engine_motion_format"] = "standard" if supported else str(reason or "missing")
        target["engine_motion_import_frame_start"] = int(frame_start)
        target["engine_motion_import_frame_end"] = int(frame_end)
        target["engine_motion_import_sample_count"] = int(sample_count)
        if signature:
            target["engine_motion_import_signature"] = signature
    if action and blob:
        action["engine_motion_original_blob"] = blob

def import_motion_as_object_transform(armature, action, motion_block, frame_start, frame_end, report=None):
    if armature or action:
        _ensure_clip_binding_id(armature, action, binding_id=_new_clip_binding_id())
    header = _read_motion_header(motion_block)
    scene = bpy.context.scene
    root_empty = _ensure_root_motion_empty(
        scene, armature, action, create=bool(motion_block), parent_armature=True
    )
    if not motion_block:
        _store_motion_metadata(armature, action, b"", False, "missing", 0, frame_start, frame_end)
        return {"result": "missing", "warning": ""}

    supported = bool(header.get("supported"))
    sample_count = int(header.get("sample_count", 0))
    if not supported:
        _store_motion_metadata(armature, action, motion_block, False, header.get("reason", "unsupported"), sample_count, frame_start, frame_end)
        _store_motion_metadata(root_empty, None, motion_block, False, header.get("reason", "unsupported"), sample_count, frame_start, frame_end)
        msg = "AnimClipMotionData preserved but could not be imported onto the root-motion Empty: unsupported motion format."
        if report:
            report({'WARNING'}, msg)
        return {"result": "unsupported", "warning": msg}

    if sample_count <= 0 or (int(header.get("flags", 0)) & ANIM_MOTION_FLAG_HAS_MOTION) == 0:
        _store_motion_metadata(armature, action, motion_block, True, "", sample_count, frame_start, frame_end)
        _store_motion_metadata(root_empty, None, motion_block, True, "", sample_count, frame_start, frame_end)
        return {"result": "flat", "warning": ""}

    count = max(1, sample_count)
    frames = _motion_sample_frames(frame_start, frame_end, count)
    samples = []
    for i in range(count):
        off = int(header["sample_start"]) + (i * ANIM_MOTION_STANDARD_SAMPLE_SIZE)
        eng_loc, eng_quat, _phase = _decode_standard_motion_sample(motion_block, off)
        bl_loc, bl_quat = engine_motion_to_blender_transform(eng_loc, eng_quat)
        samples.append((bl_loc, bl_quat))

    root_empty = _ensure_root_motion_empty(scene, armature, action, create=True, parent_armature=True)
    root_action = _write_root_motion_samples_to_object(root_empty, action, frames, samples)
    action_fcurves = list(getattr(root_action, "fcurves", [])) if root_action else []

    signature = _root_motion_signature_from_samples(sample_object_root_motion_at_frames(armature, frames))
    _store_motion_metadata(armature, action, motion_block, True, "", sample_count, frame_start, frame_end, signature)
    _store_motion_metadata(root_empty, root_action, motion_block, True, "", sample_count, frame_start, frame_end, signature)
    return {"result": "imported", "warning": "", "fcurves": action_fcurves}

def patch_motion_data_from_object(armature, action, template_block, frame_start, frame_end, options, is_looping=False):
    timeline_sample_count = max(1, int(frame_end) - int(frame_start) + 1)
    clip_motion_sample_count = max(1, int(options.get("clip_sample_count", timeline_sample_count)))
    motion_sample_count = clip_motion_sample_count + (1 if is_looping else 0)

    sample_frames = _motion_sample_frames(frame_start, frame_end, clip_motion_sample_count)
    samples_bl = sample_object_root_motion_at_frames(armature, sample_frames)
    if not samples_bl:
        identity_loc = mathutils.Vector((0.0, 0.0, 0.0))
        identity_quat = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
        samples_bl = [(identity_loc.copy(), identity_quat.copy()) for _ in range(clip_motion_sample_count)]
    while len(samples_bl) < clip_motion_sample_count:
        samples_bl.append((samples_bl[-1][0].copy(), samples_bl[-1][1].copy()))
    samples_bl = samples_bl[:clip_motion_sample_count]

    engine_mats = []
    for loc_bl, quat_bl in samples_bl:
        loc_eng, quat_eng = blender_transform_to_engine_motion(loc_bl, quat_bl)
        engine_mats.append(_matrix_from_loc_quat(loc_eng, quat_eng))

    base_mat = engine_mats[0].copy()
    base_inv = base_mat.inverted()
    rel_samples = []
    for mat in engine_mats:
        rel = base_inv @ mat
        loc, quat = _loc_quat_from_matrix(rel)
        rel_samples.append((loc, quat))

    if is_looping and rel_samples:
        loc0, quat0 = rel_samples[0]
        rel_samples.append((loc0.copy(), quat0.copy()))

    assert len(rel_samples) == motion_sample_count

    sample_bytes = bytearray()
    previous_quat = None
    for loc, quat in rel_samples:
        if previous_quat is not None and previous_quat.dot(quat) < 0.0:
            quat.negate()
        previous_quat = quat.copy()
        sample_bytes.extend(_pack_standard_motion_sample(loc, quat, 0))
    while (ANIM_MOTION_HEADER_SIZE + len(sample_bytes)) % ANIM_MOTION_BLOCK_ALIGN:
        sample_bytes.append(0)

    block = bytearray(ANIM_MOTION_HEADER_SIZE)
    struct.pack_into("<ffff", block, MOTION_INITIAL_ROT0_OFFSET, 0.0, 0.0, 0.0, 1.0)
    struct.pack_into("<ffff", block, MOTION_INITIAL_ROT1_OFFSET, 0.0, 0.0, 0.0, 1.0)
    struct.pack_into("<fff", block, MOTION_INITIAL_TRANS0_OFFSET, 0.0, 0.0, 0.0)
    struct.pack_into("<fff", block, MOTION_INITIAL_TRANS1_OFFSET, 0.0, 0.0, 0.0)

    motion_flags = ANIM_MOTION_FLAG_HAS_MOTION  # Drop phase / secondary streams: the Empty doesn't carry them.
    struct.pack_into("<BBH", block, MOTION_TYPE_OFFSET, ANIM_MOTION_TYPE_STANDARD, motion_flags, motion_sample_count)
    struct.pack_into("<IIHB5s", block, MOTION_STREAM_HEADER_OFFSET, 0, len(sample_bytes), 0, 0, b"\x00" * MOTION_HEADER_PAD_BYTES)

    abs_trans = [loc for loc, _quat in rel_samples]
    min_v = mathutils.Vector((min(v.x for v in abs_trans), min(v.y for v in abs_trans), min(v.z for v in abs_trans)))
    max_v = mathutils.Vector((max(v.x for v in abs_trans), max(v.y for v in abs_trans), max(v.z for v in abs_trans)))
    struct.pack_into("<fff", block, MOTION_BOUNDS_MIN_OFFSET, min_v.x, min_v.y, min_v.z)
    struct.pack_into("<fff", block, MOTION_BOUNDS_MAX_OFFSET, max_v.x, max_v.y, max_v.z)
    return bytes(block) + bytes(sample_bytes), ""

def _motion_block_is_flat(block):
    header = _read_motion_header(block)
    if not header.get("supported") or int(header.get("sample_count", 0)) <= 0:
        return True
    if (int(header.get("flags", 0)) & ANIM_MOTION_FLAG_HAS_MOTION) == 0:
        return True
    first_loc = None
    first_quat = None
    for i in range(int(header.get("sample_count", 0))):
        off = int(header["sample_start"]) + (i * ANIM_MOTION_STANDARD_SAMPLE_SIZE)
        loc, quat, _phase = _decode_standard_motion_sample(block, off)
        if first_loc is None:
            first_loc, first_quat = loc, quat
            continue
        if (loc - first_loc).length > 1e-5 or abs(1.0 - abs(first_quat.dot(quat))) > 1e-5:
            return False
    return True

def _required_motion_sample_count(clip_sample_count, clip_flags, motion_flags):
    count = max(1, int(clip_sample_count))
    if int(clip_flags) & FLAG_LOOPING:
        # AnimClip motion sampling can request the duplicated first sample.
        count += 1
    if int(motion_flags) & ANIM_MOTION_FLAG_HAS_SECONDARY:
        count *= 2
    return count

def validate_motion_payload_sample_count(payload, clip_sample_count, clip_flags):
    header = _read_motion_header(payload)
    if not header.get("present"):
        return False, "AnimClipMotionData block is missing."
    if not header.get("supported"):
        return True, ""
    motion_count = int(header.get("sample_count", 0))
    motion_flags = int(header.get("flags", 0))
    if motion_count <= 0 or (motion_flags & ANIM_MOTION_FLAG_HAS_MOTION) == 0:
        return True, ""
    required = _required_motion_sample_count(clip_sample_count, clip_flags, motion_flags)
    if motion_count < required:
        return False, (
            f"AnimClipMotionData has {motion_count} samples, but this export needs at least {required} "
            f"for {int(clip_sample_count)} clip samples. Re-export from the bound root-motion Empty over the same frame range."
        )
    return True, ""

def _motion_export_options(scene):
    return {}

def _root_motion_export_mode(scene):
    mode = str(getattr(scene, "engine_root_motion_export_mode", "OVERWRITE") or "OVERWRITE")
    return "INPLACE" if mode == "INPLACE" else "OVERWRITE"

def _motion_result_preview(scene, arm, action, frame_start, frame_end):
    mode = _root_motion_export_mode(scene)
    try:
        object_status = _object_motion_status(arm, frame_start, frame_end)
    except Exception:
        object_status = _object_motion_status_cached(arm, action)
    if mode == "INPLACE":
        return object_status, "Cleared"
    _source, source_error = _resolve_bound_root_motion_empty(arm, action)
    return object_status, ("Error" if source_error else "Will Overwrite")

def validate_motion_export(scene, armature, action, frame_start, frame_end, is_looping=False, clip_sample_count=None):
    mode = _root_motion_export_mode(scene)
    template = _motion_original_blob(action)
    header = _read_motion_header(template)
    original_present = bool(template)
    original_supported = bool(header.get("supported"))
    motion_clip_sample_count = max(
        1,
        int(clip_sample_count) if clip_sample_count is not None else int(frame_end) - int(frame_start) + 1,
    )
    try:
        object_status = _object_motion_status(armature, frame_start, frame_end)
    except Exception:
        object_status = "Unknown"
    warnings = []

    if mode == "INPLACE":
        if original_present and not _motion_block_is_flat(template):
            warnings.append("Original clip had root motion. Exporting in-place clears it.")
        return {
            "ok": True, "mode": mode, "result": "cleared", "payload": _motion_identity_block(),
            "warnings": warnings, "error": "", "object_status": "In-Place",
            "original_present": original_present, "original_supported": original_supported,
            "header": header,
        }

    _source, source_error = _resolve_bound_root_motion_empty(armature, action)
    if source_error:
        return {
            "ok": False, "mode": mode, "result": "error", "payload": b"",
            "warnings": warnings, "error": source_error, "object_status": "Missing Binding",
            "original_present": original_present, "original_supported": original_supported,
            "header": header,
        }
    motion_options = _motion_export_options(scene)
    motion_options["clip_sample_count"] = motion_clip_sample_count
    payload, _err = patch_motion_data_from_object(
        armature, action,
        None,
        frame_start, frame_end,
        motion_options,
        is_looping=is_looping,
    )
    patch_is_flat = _motion_block_is_flat(payload)
    if patch_is_flat:
        if original_present and not _motion_block_is_flat(template):
            warnings.append("The bound root-motion Empty is stationary; exported root motion was cleared instead of preserving the imported block.")
        result = "cleared"
        payload = _motion_identity_block()
    else:
        result = "patched"
    return {
        "ok": True, "mode": mode, "result": result, "payload": payload,
        "warnings": warnings, "error": "", "object_status": object_status,
        "original_present": original_present, "original_supported": original_supported,
        "header": header,
    }
