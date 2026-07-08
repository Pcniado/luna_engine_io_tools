# This module is part of the public-release split of blender_import_model_anim_release.py.
# Behavior-sensitive logic was moved mechanically; avoid algorithm changes here.

from .utils import *

ANIM_CLIP_BUILT_FLAGS_OFFSET = 8
TRIGGER_LOCATOR_JOINT_STRIDE = 4
TRIGGER_PAYLOAD_HEADER_SIZE = 8
TRIGGER_EVENT_OFFSET_SHIFT = 2
EVENT_DATA_ALIGN_MASK = 3
TRANS_LOG_SCALE_EPSILON = 1e-6
TRANS_LOG_SCALE_DEFAULT = 12
TRANS_LOG_SCALE_MAX = 12
TRANS_LOG_SCALE_MIN = 0
TRANS_LOG_SCALE_POWER_BIAS = 15
QUAT_PACK_SCALE = 32767
I16_MIN = -32768
I16_MAX = 32767
JOINT_PACKED_COMPONENT_COUNT = 12
SAMPLE_WORD_BYTES = 4
SAMPLE_ELEM_OFFSET_STRIDE = 4
NORMALIZED_MARKER_TIME_MAX = 65535
ANIM_CLIP_BUILT_RESERVED_BYTES = 12
DEFAULT_SOUND_EVENT_HASH = 0xDEADDEAD
DAT1_VERSION_HASH_I30 = 0x41DFFB44
DAT1_BLOCK_ALIGN = 16
STG_MAGIC = 0x00475453
STG_VERSION = 0x1
STG_HEADER_ALIGN = 16

def _skeletal_export_clear_flag_bits():
    return (
        FLAG_CURVES
        | FLAG_FACIAL_POSES
        | FLAG_PERFORMANCE
        | FLAG_CAMERA
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
    )

def _sanitize_skeletal_export_flags(flags):
    return ((int(flags) & U32_MASK) | FLAG_STD) & ~_skeletal_export_clear_flag_bits() & U32_MASK

def _unsupported_skeletal_passthrough_block_names():
    return (
        "AnimClipMorphInfo",
        "AnimClipMorphFrameData",
        "AnimClipGeomSamples",
        "AnimClipGeomSampleData",
        "AnimClipGeomMeshInfo",
        "AnimClipGeomMeshSampleData",
        "AnimClipSplineMeshInfo",
        "AnimClipSplineMeshSamples",
        "AnimClipCurves",
        "AnimClipCurvesData",
        "AnimClipPosePhonemeVisemeMap",
        "AnimClipPoseExpressionIdMap",
    )

def _drop_unsupported_skeletal_passthrough_blocks(block_payload_by_hash):
    dropped = []
    for block_name in _unsupported_skeletal_passthrough_block_names():
        block_hash = BLOCK_HASHES.get(block_name)
        if block_hash in block_payload_by_hash:
            block_payload_by_hash.pop(block_hash, None)
            dropped.append(block_name)
    return dropped

def compute_export_flags(original_flags, scene, trigger_count, motion_result, use_export_toggles=True):
    base = (int(original_flags) & U32_MASK) if original_flags is not None else FLAG_DEFAULT_EXPORT
    if use_export_toggles:
        toggles = {
            "looping": bool(getattr(scene, "engine_export_looping", False)),
            "additive": bool(getattr(scene, "engine_export_additive", False)),
            "partial": bool(getattr(scene, "engine_export_partial", False)),
            "partial_motion": bool(getattr(scene, "engine_export_partial_motion", False)),
        }
        flags = apply_flag_toggles(base, toggles)
    else:
        flags = base
    flags = _sanitize_skeletal_export_flags(flags)
    if trigger_count > 0:
        flags |= FLAG_HAS_EVENTS
    else:
        flags &= ~FLAG_HAS_EVENTS

    result = motion_result.get("result") if isinstance(motion_result, dict) else "preserved"
    if result == "preserved":
        return _sanitize_skeletal_export_flags(flags)
    if result == "patched":
        payload = motion_result.get("payload") or b""
        header = _read_motion_header(payload)
        if header.get("supported") and int(header.get("sample_count", 0)) > 0:
            flags |= FLAG_HAS_MOTION
            if int(header.get("flags", 0)) & ANIM_MOTION_FLAG_HAS_PHASE:
                flags |= FLAG_HAS_PHASE
            else:
                flags &= ~FLAG_HAS_PHASE
            flags &= ~FLAG_HAS_MOTION_SAMPLES
        return _sanitize_skeletal_export_flags(flags)
    if result == "cleared":
        flags &= ~MOTION_CLIP_FLAG_BITS
    return _sanitize_skeletal_export_flags(flags)

def _patch_anim_clip_built_flags(payload, flags):
    data = bytearray(payload or b"")
    header_off = 0
    if len(data) - header_off < ANIM_CLIP_BUILT_SIZE:
        return None, f"AnimClipBuilt block is too small: {len(data)} bytes."
    struct.pack_into("<I", data, header_off + ANIM_CLIP_BUILT_FLAGS_OFFSET, int(flags) & U32_MASK)
    return bytes(data), ""

def reconcile_export_blocks(block_payload_by_hash, clip_flags, sanitize_fn=None):
    # `sanitize_fn` lets non-skeletal exporters (camera, etc.) keep their own
    # type bits. Default preserves legacy skeletal behavior.
    if sanitize_fn is None:
        sanitize_fn = _sanitize_skeletal_export_flags
    warnings = []
    errors = []
    dropped_blocks = _drop_unsupported_skeletal_passthrough_blocks(block_payload_by_hash)
    if dropped_blocks:
        warnings.append("Dropped unsupported anim-vert/curve/facial passthrough blocks from the rebuilt skeletal AnimClip.")

    flags = sanitize_fn(clip_flags)

    built_hash = BLOCK_HASHES["AnimClipBuilt"]
    tracks_hash = BLOCK_HASHES["AnimClipTracksData"]
    trigger_hash = BLOCK_HASHES["AnimClipTriggerData"]
    marker_hash = BLOCK_HASHES["AnimClipMarkers"]
    custom_tracks_hash = BLOCK_HASHES["AnimClipCustomTracks"]
    custom_track_data_hash = BLOCK_HASHES["AnimClipCustomTrackData"]
    motion_hash = BLOCK_HASHES["AnimClipMotionData"]

    tracks_payload = block_payload_by_hash.get(tracks_hash)
    if tracks_payload is None:
        if trigger_hash in block_payload_by_hash:
            warnings.append("Dropped orphan AnimClipTriggerData block because AnimClipTracksData is missing.")
        if marker_hash in block_payload_by_hash:
            warnings.append("Dropped orphan AnimClipMarkers block because AnimClipTracksData is missing.")
        block_payload_by_hash.pop(trigger_hash, None)
        block_payload_by_hash.pop(marker_hash, None)
        flags &= ~FLAG_HAS_EVENTS
    else:
        if len(tracks_payload) < ANIM_CLIP_TRACKS_DATA_SIZE:
            warnings.append("Expanded short AnimClipTracksData header to the engine's 32-byte layout.")
        loc_count, trig_count, ev_size, marker_count, tb_data = _read_tracks_counts(tracks_payload)

        trigger_payload = block_payload_by_hash.get(trigger_hash)
        if trig_count > 0:
            trigger_joints_size = (loc_count + 1) * TRIGGER_LOCATOR_JOINT_STRIDE if loc_count else 0
            records_off = trigger_joints_size + int(ev_size)
            records_end = records_off + (trig_count * ANIM_TRIGGER_RECORD_SIZE)
            trigger_ok = (
                trigger_payload is not None
                and trigger_joints_size <= len(trigger_payload)
                and (int(ev_size) & EVENT_DATA_ALIGN_MASK) == 0
                and records_off >= trigger_joints_size
                and records_end <= len(trigger_payload)
            )
            if trigger_ok:
                event_data_end = trigger_joints_size + int(ev_size)
                for trigger_i in range(trig_count):
                    record_off = records_off + (trigger_i * ANIM_TRIGGER_RECORD_SIZE)
                    _name_hash, _loc_hash, _flags, _time, ev_off_shifted, _rad = struct.unpack_from("<IIHHHH", trigger_payload, record_off)
                    ev_off = int(ev_off_shifted) << TRIGGER_EVENT_OFFSET_SHIFT
                    if ev_off < trigger_joints_size or ev_off + TRIGGER_PAYLOAD_HEADER_SIZE > event_data_end:
                        trigger_ok = False
                        break
            if not trigger_ok:
                warnings.append("Cleared malformed animation events because TracksData counts do not fit TriggerData.")
                trig_count = 0
                ev_size = 0
                struct.pack_into("<H", tb_data, 2, 0)
                struct.pack_into("<I", tb_data, 4, 0)
                block_payload_by_hash.pop(trigger_hash, None)
        else:
            if trigger_hash in block_payload_by_hash:
                warnings.append("Dropped orphan AnimClipTriggerData block because trigger count is zero.")
            if ev_size:
                struct.pack_into("<I", tb_data, 4, 0)
                ev_size = 0
            block_payload_by_hash.pop(trigger_hash, None)

        marker_payload = block_payload_by_hash.get(marker_hash)
        if marker_count > 0:
            marker_ok = (
                marker_payload is not None
                and len(marker_payload) >= marker_count * ANIM_CLIP_MARKER_SIZE
            )
            if not marker_ok:
                warnings.append("Cleared malformed animation markers because marker count does not fit AnimClipMarkers.")
                marker_count = 0
                struct.pack_into("<H", tb_data, 8, 0)
                block_payload_by_hash.pop(marker_hash, None)
        else:
            block_payload_by_hash.pop(marker_hash, None)

        if trig_count == 0 and marker_count == 0:
            if loc_count:
                warnings.append("Dropped orphan animation locator table because there are no triggers or markers.")
            block_payload_by_hash.pop(tracks_hash, None)
            block_payload_by_hash.pop(trigger_hash, None)
            block_payload_by_hash.pop(marker_hash, None)
        else:
            block_payload_by_hash[tracks_hash] = bytes(tb_data)

        if trig_count > 0 and trigger_hash in block_payload_by_hash:
            flags |= FLAG_HAS_EVENTS
        else:
            flags &= ~FLAG_HAS_EVENTS

    custom_tracks_payload = block_payload_by_hash.get(custom_tracks_hash)
    custom_track_count = 0
    if custom_tracks_payload is not None:
        custom_track_count = len(custom_tracks_payload) // ANIM_CURVE_BUILT_SIZE
        has_remainder = (len(custom_tracks_payload) % ANIM_CURVE_BUILT_SIZE) != 0
        has_data_block = custom_track_data_hash in block_payload_by_hash
        if custom_track_count == 0 or has_remainder or not has_data_block:
            warnings.append("Dropped malformed custom track blocks so kAnimFlagsHasCustomTracks matches engine load state.")
            custom_track_count = 0
            block_payload_by_hash.pop(custom_tracks_hash, None)
            block_payload_by_hash.pop(custom_track_data_hash, None)
    elif custom_track_data_hash in block_payload_by_hash:
        warnings.append("Dropped orphan AnimClipCustomTrackData block because AnimClipCustomTracks is missing.")
        block_payload_by_hash.pop(custom_track_data_hash, None)

    if custom_track_count > 0:
        flags |= FLAG_HAS_CUSTOM_TRACKS
    else:
        flags &= ~FLAG_HAS_CUSTOM_TRACKS

    motion_payload = block_payload_by_hash.get(motion_hash)
    if motion_payload is None:
        if not (flags & (FLAG_CURVES | FLAG_PERFORMANCE)):
            errors.append("AnimClipMotionData block is missing for a non-curve/non-performance clip.")
    else:
        motion_header = _read_motion_header(motion_payload)
        if int(motion_header.get("sample_count", 0)) > 0 and not (flags & (FLAG_HAS_MOTION | FLAG_HAS_PHASE)):
            warnings.append("Enabled kAnimFlagsHasMotion because MotionData has samples.")
            flags |= FLAG_HAS_MOTION

    built_payload = block_payload_by_hash.get(built_hash)
    if built_payload is None:
        errors.append("AnimClipBuilt block is missing.")
    else:
        patched_built, error = _patch_anim_clip_built_flags(built_payload, flags)
        if error:
            errors.append(error)
        else:
            block_payload_by_hash[built_hash] = patched_built

    return flags & U32_MASK, warnings, errors

def _compute_trans_log_scale(max_abs_trans):
    import math
    if max_abs_trans < TRANS_LOG_SCALE_EPSILON:
        return TRANS_LOG_SCALE_DEFAULT

    rounded = 1
    target = max(1, math.ceil(max_abs_trans))
    while rounded < target:
        rounded <<= 1
    log2_pow2 = rounded.bit_length() - 1
    log_scale = TRANS_LOG_SCALE_POWER_BIAS - log2_pow2
    return max(TRANS_LOG_SCALE_MIN, min(TRANS_LOG_SCALE_MAX, log_scale))

def _quat_dot(a, b):
    return (a.w * b.w) + (a.x * b.x) + (a.y * b.y) + (a.z * b.z)

def _negate_quat(q):
    q.w = -q.w
    q.x = -q.x
    q.y = -q.y
    q.z = -q.z
    return q

def _continuous_quat_loc_samples(all_locals, joint_count):
    prev_rots = [None] * joint_count
    samples = []
    for frame in all_locals:
        row = []
        for j, local_m in enumerate(frame):
            loc, rot, _ = local_m.decompose()
            rot.normalize()
            if prev_rots[j] is not None and _quat_dot(prev_rots[j], rot) < 0.0:
                _negate_quat(rot)
            prev_rots[j] = rot.copy()
            row.append((loc.copy(), rot.copy()))
        samples.append(row)
    return samples

def _quantize_quat_trans_values(loc, rot, trans_shift):
    qx = int(round(rot.x * QUAT_PACK_SCALE))
    qy = int(round(rot.y * QUAT_PACK_SCALE))
    qz = int(round(rot.z * QUAT_PACK_SCALE))
    qw = int(round(rot.w * QUAT_PACK_SCALE))

    t_div = 1 << trans_shift
    tx = int(round(loc.x * t_div))
    ty = int(round(loc.y * t_div))
    tz = int(round(loc.z * t_div))

    def clamp_i16(v):
        return max(I16_MIN, min(I16_MAX, v))

    return (
        0, 0, 0, 0,
        clamp_i16(qx), clamp_i16(qy), clamp_i16(qz), clamp_i16(qw),
        clamp_i16(tx), clamp_i16(ty), clamp_i16(tz), trans_shift,
    )

def _quantize_quat_trans(local_m, trans_shift):
    loc, rot, _ = local_m.decompose()
    rot.normalize()
    return _quantize_quat_trans_values(loc, rot, trans_shift)

def _compute_frame_engine_locals(arm, joint_names, parent_map, frame):
    bpy.context.scene.frame_set(frame)
    n = len(joint_names)
    target = [None] * n
    local = [None] * n
    swz_inv = SWIZZLE_MAT.inverted()
    IDENTITY = mathutils.Matrix.Identity(4)

    def resolve(start):

        stack = []
        cur = start
        while cur != -1 and target[cur] is None:
            stack.append(cur)
            cur = parent_map[cur]

        while stack:
            i = stack.pop()
            pb = arm.pose.bones.get(joint_names[i])
            if pb is None:
                target[i] = IDENTITY
                local[i] = IDENTITY
                continue
            basis = pb.matrix_basis
            rest_local = pb.bone.matrix_local
            p = parent_map[i]
            if p == -1:
                target[i] = rest_local @ basis
                local[i] = swz_inv @ target[i]
            else:
                parent_pb = arm.pose.bones.get(joint_names[p])
                if parent_pb is None or target[p] is None:

                    target[i] = rest_local @ basis
                    local[i] = swz_inv @ target[i]
                else:
                    parent_rest = parent_pb.bone.matrix_local.inverted() @ rest_local
                    target[i] = target[p] @ parent_rest @ basis
                    local[i] = target[p].inverted() @ target[i]

    for i in range(n):
        if target[i] is None:
            resolve(i)
    return local

def _idprop_int(container, key, default=0):
    if not container:
        return default
    try:
        return int(container.get(key, default))
    except Exception:
        return default

def _idprop_float(container, key, default=0.0):
    if not container:
        return default
    try:
        return float(container.get(key, default))
    except Exception:
        return default

def _resolve_export_timing(arm, action, resident_sample_count, resident_fps, original_flags=0):
    resident_sample_count = max(1, int(resident_sample_count))
    resident_fps = float(resident_fps) if resident_fps > 0.0 else 30.0
    clip_sample_count = resident_sample_count
    clip_fps = resident_fps

    for target in (action, arm):
        full_count = _idprop_int(target, "engine_clip_full_sample_count", 0)
        full_fps = _idprop_float(target, "engine_clip_full_fps", 0.0)
        imported_count = _idprop_int(target, "engine_clip_import_sample_count", 0)
        stored_resident_count = _idprop_int(target, "engine_clip_resident_sample_count", 0)
        imported_fps = _idprop_float(target, "engine_clip_import_fps", 0.0)
        stored_resident_fps = _idprop_float(target, "engine_clip_resident_fps", 0.0)
        matching_count = resident_sample_count in {
            count for count in (imported_count, stored_resident_count) if count > 0
        }
        matching_rate = any(
            abs(resident_fps - rate) <= 0.01
            for rate in (imported_fps, stored_resident_fps)
            if rate > 0.0
        )
        if full_count > 0 and full_fps > 0.0 and matching_count and matching_rate:
            clip_sample_count = full_count
            clip_fps = full_fps
            break

    if clip_sample_count == resident_sample_count:
        template = _motion_original_blob(action)
        header = _read_motion_header(template)
        motion_count = int(header.get("sample_count", 0)) if header.get("supported") else 0
        is_looping = bool(int(original_flags) & FLAG_LOOPING)
        full_count = motion_count - (1 if is_looping and motion_count > 1 else 0)
        if full_count > resident_sample_count:
            resident_duration = _duration_from_sample_timing(resident_sample_count, resident_fps, is_looping)
            full_span = full_count if is_looping else max(1, full_count - 1)
            inferred_fps = (float(full_span) / resident_duration) if resident_duration > 0.0 else 0.0
            if inferred_fps > 0.0:
                clip_sample_count = full_count
                clip_fps = inferred_fps

    return clip_sample_count, clip_fps, resident_sample_count, resident_fps

def _duration_from_sample_timing(sample_count, fps, is_looping):
    sample_count = max(1, int(sample_count))
    fps = float(fps)
    if fps <= 0.0:
        fps = 30.0
    span = sample_count if is_looping else max(0, sample_count - 1)
    return span / fps

class ExportEngineAnim(Operator, ExportHelper):
    bl_idname = "export_anim.engine_anim"
    bl_label = "Export Luna Engine Anim"
    bl_description = "Export the selected armature or camera animation as a .animclip"
    bl_options = {'REGISTER', 'PRESET'}
    filename_ext = ".animclip"
    filter_glob: StringProperty(default="*.animclip", options={'HIDDEN'})

    @staticmethod
    def _camera_has_animation_data(camera):
        if not camera:
            return False
        anim_data = getattr(camera, "animation_data", None)
        if anim_data and getattr(anim_data, "action", None):
            return True
        cam_data = getattr(camera, "data", None)
        data_anim = getattr(cam_data, "animation_data", None) if cam_data else None
        return bool(data_anim and getattr(data_anim, "action", None))

    def _resolve_export_target(self, context):
        if "sync_luna_settings_for_context" in globals():
            try:
                sync_luna_settings_for_context(context, force=False)
            except Exception:
                pass

        active = getattr(context, "object", None) or getattr(context, "active_object", None)
        selected = list(getattr(context, "selected_objects", []) or [])
        if not active and not selected:
            return None, "Nothing is selected. Select the armature, mesh child, root-motion Empty, or camera you want to export."

        if active and getattr(active, "type", None) == 'CAMERA':
            if not self._camera_has_animation_data(active):
                return None, f"'{active.name}' has no camera or lens animation action. Assign an action before exporting."
            return 'camera', active

        arm = _resolve_anim_armature(context)
        if arm and getattr(arm, "type", None) == 'ARMATURE':
            if not arm.animation_data or not arm.animation_data.action:
                return None, f"'{arm.name}' has no active action. Assign an action before exporting."
            return 'armature', arm

        camera = _resolve_anim_camera(context)
        if camera:
            if not self._camera_has_animation_data(camera):
                return None, f"'{camera.name}' has no camera or lens animation action. Assign an action before exporting."
            return 'camera', camera

        names = ", ".join(obj.name for obj in selected[:3]) if selected else (active.name if active else "")
        suffix = f" (selected: {names})" if names else ""
        return None, (
            "Select an armature, mesh child, root-motion Empty, or camera with animation data"
            f" before exporting.{suffix}"
        )

    def invoke(self, context, event):
        kind, payload = self._resolve_export_target(context)
        if kind is None:
            show_popup("Cannot Export AnimClip", payload, icon='ERROR')
            self.report({'ERROR'}, payload)
            return {'CANCELLED'}
        return super().invoke(context, event)

    def execute(self, context):
        kind, payload = self._resolve_export_target(context)
        if kind is None:
            self.report({'ERROR'}, payload)
            return {'CANCELLED'}

        wm = context.window_manager
        wm.progress_begin(0, 100)
        try:
            if kind == 'camera':
                return export_camera_animclip(context, self, payload, self.filepath)
            return self._do_export(context, payload)
        finally:
            wm.progress_end()

    def _do_export(self, context, arm):
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
        sample_cnt = max(1, frame_end - frame_start + 1)
        fps = float(getattr(scene, "engine_anim_fps", scene.get("engine_base_fps", 0.0)))
        if fps <= 0.0:
            fps = scene.render.fps / max(1e-6, scene.render.fps_base)
        if fps <= 0.0:
            self.report({'ERROR'}, "Export FPS must be greater than zero.")
            scene.frame_set(original_frame)
            return {'CANCELLED'}

        bone_hash_to_name = {string_crc32(b.name): b.name for b in arm.data.bones}
        stored_hashes = arm.get("engine_clip_joint_hashes")
        stored_jc = arm.get("engine_clip_joint_count")

        export_joint_hashes = None
        if stored_hashes and stored_jc:
            joint_count = int(stored_jc)
            full_hashes = list(stored_hashes)
            export_joint_hashes = [int(h) & U32_MASK for h in full_hashes[:joint_count]]

            # Preserve imported joint hash order; missing Blender bones stay as
            # synthetic Joint_XXXXXXXX names so the binary tables remain stable.
            joint_names = []
            for jh in full_hashes[:joint_count]:
                name = bone_hash_to_name.get(int(jh) & U32_MASK)
                if name is None:
                    name = f"Joint_{int(jh) & U32_MASK:08X}"
                joint_names.append(name)

            export_hash_tail = [int(h) & U32_MASK for h in full_hashes[joint_count:]]
        else:
            joint_names = [b.name for b in arm.data.bones]
            joint_count = len(joint_names)
            export_hash_tail = []

        name_to_idx = {n: i for i, n in enumerate(joint_names)}
        parent_map = [-1] * joint_count
        for i, name in enumerate(joint_names):
            pb = arm.pose.bones.get(name)
            if pb and pb.parent and pb.parent.name in name_to_idx:
                parent_map[i] = name_to_idx[pb.parent.name]

        all_locals = []

        for fi in range(sample_cnt):
            if fi % max(1, sample_cnt // 50) == 0:
                context.window_manager.progress_update(int(25 * fi / sample_cnt))
            all_locals.append(_compute_frame_engine_locals(arm, joint_names, parent_map, frame_start + fi))
        continuous_samples = _continuous_quat_loc_samples(all_locals, joint_count)

        per_joint_trans_shift = []
        for j in range(joint_count):
            max_abs = 0.0
            for fi in range(sample_cnt):
                loc, _ = continuous_samples[fi][j]
                m = max(abs(loc.x), abs(loc.y), abs(loc.z))
                if m > max_abs:
                    max_abs = m
            per_joint_trans_shift.append(_compute_trans_log_scale(max_abs))

        frame_values = []
        for fi in range(sample_cnt):
            if fi % max(1, sample_cnt // 50) == 0:
                context.window_manager.progress_update(25 + int(25 * fi / sample_cnt))
            row = []
            for j in range(joint_count):
                loc, rot = continuous_samples[fi][j]
                row.append(_quantize_quat_trans_values(loc, rot, per_joint_trans_shift[j]))
            frame_values.append(row)

        base_pose = [[0] * JOINT_PACKED_COMPONENT_COUNT for _ in range(joint_count)]
        max_delta = [[0] * JOINT_PACKED_COMPONENT_COUNT for _ in range(joint_count)]
        for j in range(joint_count):
            for c in range(JOINT_PACKED_COMPONENT_COUNT):
                values = [frame_values[f][j][c] for f in range(sample_cnt)]
                mn = min(values)
                mx = max(values)
                base_pose[j][c] = mn
                max_delta[j][c] = mx - mn

        ANIMATED_SLOTS = (4, 5, 6, 7, 8, 9, 10)
        elem_table = []
        large_table = []
        for j in range(joint_count):
            for c in ANIMATED_SLOTS:
                d = max_delta[j][c]
                if d == 0:
                    continue
                bits = d.bit_length()
                if bits > 16:
                    large_table.append((j, c, bits))
                else:
                    elem_table.append((j, c, bits))

        bs = BitStreamWriter()
        bits_per_frame = sum(b for _, _, b in elem_table) + sum(b for _, _, b in large_table)
        stride_bits = (bits_per_frame + 31) & ~31
        stride = stride_bits // 8

        for fi in range(sample_cnt):
            if fi % max(1, sample_cnt // 50) == 0:
                context.window_manager.progress_update(40 + int(40 * fi / sample_cnt))
            for j, c, bits in elem_table:
                delta = frame_values[fi][j][c] - base_pose[j][c]
                bs.write_bits(delta, bits)
            for j, c, bits in large_table:
                delta = frame_values[fi][j][c] - base_pose[j][c]
                bs.write_bits(delta, bits)
            bs.reset_frame()

        total_words_expected = sample_cnt * (stride // SAMPLE_WORD_BYTES)
        while len(bs.words) < total_words_expected:
            bs.words.append(0)
        sample_data = struct.pack(f"<{len(bs.words)}I", *bs.words)

        use_original_values = bool(getattr(scene, "engine_export_use_original_values", True))
        action = arm.animation_data.action if arm.animation_data else None
        migrate_clip_flags(arm, action)
        original_flags = get_original_flags(action, arm)
        if original_flags is None:
            original_flags = FLAG_DEFAULT_EXPORT

        clip_sample_count_for_timing = int(sample_cnt)
        clip_fps_for_timing = float(fps)
        resident_sample_count = clip_sample_count_for_timing
        resident_fps = clip_fps_for_timing

        pt_dict_for_export = {}
        if action and "engine_passthrough_blocks" in action:
            try:
                pt_dict_for_export = json.loads(str(action["engine_passthrough_blocks"]))
            except Exception:
                pass

        trigger_count_for_flags = len(_action_trigger_marker_entries(action)) if action else 0
        # Need the looping bit before motion patching so the motion block carries
        # sample_count + 1 samples
        intent_flags = compute_export_flags(
            original_flags,
            scene,
            trigger_count_for_flags,
            None,
            use_export_toggles=True,
        )
        is_looping_intent = bool(intent_flags & FLAG_LOOPING)
        motion_export = validate_motion_export(
            scene, arm, action, frame_start, frame_end,
            is_looping=is_looping_intent,
            clip_sample_count=clip_sample_count_for_timing,
        )
        if not motion_export.get("ok"):
            self.report({'ERROR'}, motion_export.get("error", "Root motion export failed."))
            scene.frame_set(original_frame)
            return {'CANCELLED'}
        for warning in motion_export.get("warnings", []):
            self.report({'WARNING'}, warning)

        clip_flags = compute_export_flags(
            original_flags,
            scene,
            trigger_count_for_flags,
            motion_export,
            use_export_toggles=True,
        )
        anim_clip_motion_data = motion_export.get("payload") or _motion_identity_block()

        motion_count_ok, motion_count_error = validate_motion_payload_sample_count(
            anim_clip_motion_data,
            clip_sample_count_for_timing,
            clip_flags,
        )
        if not motion_count_ok:
            self.report({'ERROR'}, motion_count_error)
            scene.frame_set(original_frame)
            return {'CANCELLED'}
        set_current_flags(arm, clip_flags)
        if action:
            set_current_flags(action, clip_flags)

        want_looping = bool(clip_flags & FLAG_LOOPING)
        want_additive = bool(clip_flags & FLAG_IS_ADDITIVE)
        want_partial = bool(clip_flags & FLAG_IS_PARTIAL)
        want_partial_motion = bool(clip_flags & FLAG_PARTIAL_MOTION)

        duration = _duration_from_sample_timing(clip_sample_count_for_timing, clip_fps_for_timing, want_looping)

        clip_name = os.path.splitext(os.path.basename(self.filepath))[0]
        stored_name_hash = arm.get("engine_clip_name_hash")
        if use_original_values and stored_name_hash is not None:
            clip_name_hash = int(stored_name_hash) & U32_MASK
        else:
            clip_name_hash = string_crc32(clip_name)
        cycle_count = int(arm.get("engine_clip_cycle_count", 1)) & 0xFF if use_original_values else 1
        facial_name_hash = int(arm.get("engine_clip_facial_name_hash", 0)) & U32_MASK if use_original_values else 0
        sound_event_hash = int(arm.get("engine_clip_sound_event_hash", DEFAULT_SOUND_EVENT_HASH)) & U32_MASK if use_original_values else DEFAULT_SOUND_EVENT_HASH
        particle_count_max = int(arm.get("engine_clip_particle_count_max", 0)) & U32_MASK if use_original_values else 0

        header = struct.pack( # animclipbuilt struct
            "<IIIffBBBBII HH I I HH HH HH HH HH f I I I I 12s".replace(" ", ""),
            0,

            clip_name_hash,

            clip_flags,

            clip_fps_for_timing,

            duration,

            cycle_count, 0, 0, 0,

            0,

            0,

            joint_count,

            clip_sample_count_for_timing,

            len(elem_table),

            stride,

            0,

            0,

            clip_sample_count_for_timing,

            0,

            0,

            0,

            0,

            0,

            0,

            resident_sample_count,

            resident_fps,

            len(large_table),

            facial_name_hash,

            sound_event_hash,

            particle_count_max,

            b'\x00' * ANIM_CLIP_BUILT_RESERVED_BYTES,

        )
        anim_clip_built = header
        assert len(anim_clip_built) == 96, f"Header size wrong: {len(anim_clip_built)}"

        SCALE_LOG_DEFAULT = 12
        base_state_parts = []
        for j in range(joint_count):
            bp = base_pose[j]
            base_state_parts.append(struct.pack(
                "<hhhhhhhBB",
                bp[4], bp[5], bp[6], bp[7],
                bp[8], bp[9], bp[10],
                SCALE_LOG_DEFAULT,
                per_joint_trans_shift[j],
            ))
        anim_clip_base_state = b''.join(base_state_parts)

        elem_offs_bytes = b''.join(struct.pack("<I", (j * JOINT_PACKED_COMPONENT_COUNT + c) * SAMPLE_ELEM_OFFSET_STRIDE) for j, c, _ in elem_table)
        elem_strs_bytes = bytes(((b - 1) & 0xF) for _, _, b in elem_table)

        large_start = (len(elem_offs_bytes) + len(elem_strs_bytes) + 3) & ~3
        padding_elem = b'\x00' * (large_start - (len(elem_offs_bytes) + len(elem_strs_bytes)))
        large_offs_bytes = b''.join(struct.pack("<I", (j * JOINT_PACKED_COMPONENT_COUNT + c) * SAMPLE_ELEM_OFFSET_STRIDE) for j, c, _ in large_table)

        large_strs_bytes = b''.join(struct.pack("<H", (b - 1) & 0xFF) for _, _, b in large_table)
        anim_clip_sample_elem = (
            elem_offs_bytes + elem_strs_bytes + padding_elem
            + large_offs_bytes + large_strs_bytes
        )

        anim_clip_sample_data = sample_data

        if export_joint_hashes is None:
            export_joint_hashes = [string_crc32(n) for n in joint_names]
        per_joint_hashes = b''.join(struct.pack("<I", h) for h in export_joint_hashes)
        tail_hashes = b''.join(struct.pack("<I", h) for h in export_hash_tail)
        anim_clip_joint_hashes = per_joint_hashes + tail_hashes

        # anim_clip_motion_data is already determined above based on use_original_values.

        out_blocks = [
            (BLOCK_HASHES["AnimClipBuilt"],            anim_clip_built),
            (BLOCK_HASHES["AnimClipBaseState"],        anim_clip_base_state),
            (BLOCK_HASHES["AnimClipSampleElem"],       anim_clip_sample_elem),
            (BLOCK_HASHES["AnimClipSampleDataResident"], anim_clip_sample_data),
            (BLOCK_HASHES["AnimClipJointHashes"],      anim_clip_joint_hashes),
            (BLOCK_HASHES["AnimClipMotionData"],       anim_clip_motion_data),
        ]

        import base64
        import re
        if action and (pt_dict_for_export or "engine_passthrough_blocks" in action or trigger_count_for_flags > 0):
            try:
                pt_dict = dict(pt_dict_for_export)
                if not pt_dict and "engine_passthrough_blocks" in action:
                    pt_dict = json.loads(str(action["engine_passthrough_blocks"]))
                pt_dict[str(BLOCK_HASHES["AnimClipMotionData"])] = base64.b64encode(anim_clip_motion_data).decode('ascii')
                

                h_str_tracks = str(BLOCK_HASHES["AnimClipTracksData"])
                h_str_trigger = str(BLOCK_HASHES["AnimClipTriggerData"])
                h_str_marker = str(BLOCK_HASHES["AnimClipMarkers"])

                trigger_entries = _action_trigger_marker_entries(action)
                has_pose_markers = any(
                    re.match(r"Marker_\[(\d+)\]_", m.name)
                    for m in (action.pose_markers if action else [])
                )
                if not trigger_entries and not has_pose_markers:
                    pt_dict.pop(h_str_tracks, None)
                    pt_dict.pop(h_str_trigger, None)
                    pt_dict.pop(h_str_marker, None)
                if trigger_entries and h_str_tracks not in pt_dict:
                    pt_dict[h_str_tracks] = base64.b64encode(_empty_tracks_data_block()).decode('ascii')

                if h_str_tracks in pt_dict:
                    loc_count, trig_count, ev_size, marker_count, tb_data = _read_tracks_counts(base64.b64decode(pt_dict[h_str_tracks]))
                    
                    total_frames = max(1, sample_cnt - 1)
                    

                    old_trig_count = trig_count
                    trig_count = len(trigger_entries)
                    struct.pack_into("<H", tb_data, 2, trig_count)

                    if trig_count > 0:
                        tg_data = bytearray(base64.b64decode(pt_dict.get(h_str_trigger, "")))
                        tg_data, tb_data = rebuild_trigger_data_block(
                            bytes(tg_data), bytes(tb_data), loc_count, trig_count, action,
                            total_frames=total_frames, old_trig_count=old_trig_count,
                            frame_start=frame_start,
                        )
                        pt_dict[h_str_trigger] = base64.b64encode(tg_data).decode('ascii')
                        pt_dict[h_str_tracks] = base64.b64encode(tb_data).decode('ascii')
                    else:
                        pt_dict.pop(h_str_trigger, None)
                        struct.pack_into("<I", tb_data, 4, 0)
                        pt_dict[h_str_tracks] = base64.b64encode(tb_data).decode('ascii')
                        

                    if h_str_marker in pt_dict and marker_count > 0:
                        mk_data = bytearray(base64.b64decode(pt_dict[h_str_marker]))
                        if len(mk_data) >= marker_count * ANIM_CLIP_MARKER_SIZE:
                            for marker in action.pose_markers:
                                match = re.match(r"Marker_\[(\d+)\]_([0-9a-fA-F]+)", marker.name)
                                if match:
                                    idx = int(match.group(1))
                                    if idx < marker_count:

                                        start_frame = max(0, min(NORMALIZED_MARKER_TIME_MAX, marker.frame))
                                        struct.pack_into("<H", mk_data, idx * ANIM_CLIP_MARKER_SIZE + 2, start_frame)
                        pt_dict[h_str_marker] = base64.b64encode(mk_data).decode('ascii')
                    elif marker_count > 0:
                        marker_count = 0
                        struct.pack_into("<H", tb_data, 8, 0)
                        pt_dict[h_str_tracks] = base64.b64encode(tb_data).decode('ascii')

                    loc_count, trig_count, ev_size, marker_count, tb_data = _read_tracks_counts(tb_data)
                    if trig_count == 0 and marker_count == 0:
                        pt_dict.pop(h_str_tracks, None)
                        pt_dict.pop(h_str_trigger, None)
                        pt_dict.pop(h_str_marker, None)
                    elif marker_count == 0:
                        pt_dict.pop(h_str_marker, None)
                else:
                    pt_dict.pop(h_str_trigger, None)
                    pt_dict.pop(h_str_marker, None)

                for h_str, b64_data in pt_dict.items():
                    out_blocks.append((int(h_str), base64.b64decode(b64_data)))
            except Exception as e:
                log_exception("Failed to load or patch AnimClip passthrough blocks")
                self.report({'WARNING'}, f"Could not patch preserved AnimClip passthrough blocks: {e}")

        block_payload_by_hash = {}
        for block_hash, payload in out_blocks:
            block_payload_by_hash[block_hash] = payload
        sanitize_fn = None
        clip_flags, export_warnings, export_errors = reconcile_export_blocks(
            block_payload_by_hash, clip_flags, sanitize_fn=sanitize_fn
        )
        for warning in export_warnings:
            self.report({'WARNING'}, warning)
        if export_errors:
            self.report({'ERROR'}, " ".join(export_errors))
            scene.frame_set(original_frame)
            return {'CANCELLED'}
        set_current_flags(arm, clip_flags)
        if action:
            set_current_flags(action, clip_flags)
        # DAT1 block order and alignment are part of the observed engine layout.
        blocks = sorted(block_payload_by_hash.items(), key=lambda b: b[0])

        block_count = len(blocks)
        fixup_count = 0

        dat1_header_size = (
            DAT1_HEADER_SIZE
            + block_count * DAT1_BLOCK_TABLE_ENTRY_SIZE
            + fixup_count * DAT1_FIXUP_TABLE_ENTRY_SIZE
        )

        block_offsets = []
        cursor = dat1_header_size

        block_payload_parts = []
        for _, payload in blocks:
            aligned = (cursor + DAT1_BLOCK_ALIGN - 1) & ~(DAT1_BLOCK_ALIGN - 1)
            pad = aligned - cursor
            if pad:
                block_payload_parts.append(b'\x00' * pad)
                cursor = aligned
            block_offsets.append(cursor)
            block_payload_parts.append(payload)
            cursor += len(payload)

        block_table = b''.join(
            struct.pack("<III", h, off, len(p)) for (h, p), off in zip(blocks, block_offsets)
        )
        fixup_table = b''

        body = b''.join(block_payload_parts)

        chunk_size = DAT1_HEADER_SIZE + len(block_table) + len(fixup_table) + len(body)
        dat1_header = struct.pack(
            "<IIIHH",
            DAT1_FILE_ID,
            DAT1_VERSION_HASH_I30,
            chunk_size,
            block_count,
            fixup_count,
        )

        out = dat1_header + block_table + fixup_table + body

        asset_header = struct.pack("<IBBHII",
            DAT1_VERSION_HASH_I30,
            0,
            1,
            0,
            chunk_size,
            0,
        )

        stg_buf = bytearray()

        stg_buf += struct.pack("<IIII",
            STG_MAGIC,
            STG_VERSION,
            len(asset_header),
            0,
        )

        stg_buf += asset_header

        rem = len(stg_buf) % STG_HEADER_ALIGN
        if rem != 0:
            stg_buf += b'\x00' * (STG_HEADER_ALIGN - rem)

        add_stg_header = bool(getattr(scene, "engine_export_add_stg_header", True))
        final_out = (bytes(stg_buf) + out) if add_stg_header else out

        try:
            with open(self.filepath, "wb") as f:
                f.write(final_out)
        except OSError as e:
            self.report({'ERROR'}, f"Write failed: {e}")
            scene.frame_set(original_frame)
            return {'CANCELLED'}

        scene.frame_set(original_frame)
        context.window_manager.progress_update(100)
        format_name = "STG+DAT1" if add_stg_header else "raw DAT1"
        self.report(
            {'INFO'},
            f"Wrote {format_name}: {sample_cnt} frames, {joint_count} joints, {len(elem_table)} elems / {len(large_table)} large, stride={stride}B"
        )
        return {'FINISHED'}
