# This module is part of the public-release split of blender_import_model_anim_release.py.
# Behavior-sensitive logic was moved mechanically; avoid algorithm changes here.

from .utils import *

CLIP_BUILT_FLAGS_INDEX = 2
CLIP_BUILT_NAME_HASH_INDEX = 1
CLIP_BUILT_CYCLE_COUNT_INDEX = 5
CLIP_BUILT_PAGED_FPS_INDEX = 3
CLIP_BUILT_DURATION_INDEX = 4
CLIP_BUILT_PAGED_SAMPLE_COUNT_INDEX = 12
CLIP_BUILT_UNIQUE_SAMPLE_COUNT_INDEX = 17
CLIP_BUILT_RESIDENT_SAMPLE_COUNT_INDEX = 24
CLIP_BUILT_RESIDENT_FPS_INDEX = 25
CLIP_BUILT_FACIAL_NAME_HASH_INDEX = 27
CLIP_BUILT_SOUND_EVENT_HASH_INDEX = 28
CLIP_BUILT_PARTICLE_COUNT_MAX_INDEX = 29
CLIP_BUILT_JOINT_COUNT_INDEX = 11
CLIP_BUILT_ELEM_COUNT_INDEX = 13
CLIP_BUILT_STRIDE_INDEX = 14
CLIP_BUILT_LARGE_ELEM_COUNT_INDEX = 26
DEFAULT_IMPORT_FPS = 30.0
JOINT_PACKED_COMPONENT_COUNT = 12
SAMPLE_ELEM_OFFSET_STRIDE = 4
SAMPLE_WORD_BYTES = 4
BITS_PER_BYTE = 8
SAMPLE_TRAILING_PAD_BYTES = 16
SAMPLE_LOOKUP_ALIGN = 16
U16_FIELD_BYTES = 2
U32_FIELD_BYTES = 4
BASE_STATE_SHORTS_PER_JOINT = 8
ROTATION_COMPONENT_COUNT = 4
LOCATION_COMPONENT_COUNT = 3
NORMALIZED_TRIGGER_TIME_MAX = 65535
TRIGGER_LOCATOR_JOINT_STRIDE = 4
TRIGGER_PAYLOAD_HEADER_SIZE = 8

class ImportEngineAnim(Operator, ImportHelper):
    bl_idname = "import_anim.engine_anim"
    bl_label = "Import Luna Engine Anim"
    bl_description = "Import one or more engine-compiled AnimClip (.animclip) files. Select an armature for skeletal clips or a camera for camera clips"
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}
    filename_ext = ".animclip"
    filter_glob: StringProperty(default="*.animclip;*.dat1", options={'HIDDEN'})
    files: CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        return super().invoke(context, event)

    def _resolve_filepaths(self):
        paths = []
        selected_files = list(getattr(self, "files", []) or [])
        if selected_files:
            base = self.directory or os.path.dirname(self.filepath or "")
            for entry in selected_files:
                name = getattr(entry, "name", "") or ""
                if not name:
                    continue
                paths.append(os.path.join(base, name))
        if not paths and self.filepath:
            paths.append(self.filepath)

        unique_paths = []
        seen = set()
        for path in paths:
            if not path:
                continue
            key = os.path.normcase(os.path.abspath(path))
            if key in seen:
                continue
            seen.add(key)
            unique_paths.append(path)
        return unique_paths

    def execute(self, context):
        filepaths = self._resolve_filepaths()
        if not filepaths:
            self.report({'ERROR'}, "No file selected. Pick at least one .animclip file to import.")
            return {'CANCELLED'}

        wm = context.window_manager
        wm.progress_begin(0, 100)
        success_count = 0
        failure_messages = []
        try:
            total = len(filepaths)
            for index, filepath in enumerate(filepaths):
                if not os.path.isfile(filepath):
                    failure_messages.append(f"{os.path.basename(filepath) or filepath}: file not found")
                    continue
                try:
                    result = self._do_import(context, wm, filepath)
                except Exception as exc:
                    log_exception("Unexpected error importing AnimClip %s", filepath)
                    failure_messages.append(f"{os.path.basename(filepath)}: {exc}")
                    continue
                if result == {'FINISHED'}:
                    success_count += 1
                else:
                    failure_messages.append(f"{os.path.basename(filepath)}: import cancelled")
                if total > 1:
                    wm.progress_update(int(100 * ((index + 1) / total)))
        finally:
            wm.progress_end()

        if success_count == 0:
            detail = "; ".join(failure_messages) if failure_messages else "no clips imported"
            self.report({'ERROR'}, f"AnimClip import failed: {detail}")
            return {'CANCELLED'}

        if failure_messages:
            self.report({'WARNING'}, f"Imported {success_count}/{len(filepaths)} clips. Skipped: {'; '.join(failure_messages)}")
        else:
            self.report({'INFO'}, f"Imported {success_count} clip{'s' if success_count != 1 else ''}.")
        return {'FINISHED'}

    def _do_import(self, context, wm, filepath):
        data, blocks, table_end = get_dat1_data(filepath)
        if not data:
            self.report({'ERROR'}, f"{os.path.basename(filepath)}: invalid DAT1 magic signature not found.")
            return {'CANCELLED'}
        if BLOCK_HASHES["AnimClipBuilt"] not in blocks:
            self.report({'ERROR'}, f"{os.path.basename(filepath)}: file does not contain Animation data.")
            return {'CANCELLED'}

        wm.progress_update(2)

        built_off, built_size = blocks[BLOCK_HASHES["AnimClipBuilt"]]
        



        header_off = 0
        if built_size - header_off < ANIM_CLIP_BUILT_SIZE:
            self.report({'ERROR'}, f"AnimClipBuilt block is too small: {built_size} bytes.")
            return {'CANCELLED'}

        cb = struct.unpack_from(ANIM_CLIP_BUILT_FORMAT, data, built_off + header_off)

        _debug_print(f"DEBUG AnimClipBuilt Header (offset {header_off}): {cb[:30]}")
        
        clip_flags = cb[CLIP_BUILT_FLAGS_INDEX]
        anim_fps_paged = cb[CLIP_BUILT_PAGED_FPS_INDEX]
        anim_duration = cb[CLIP_BUILT_DURATION_INDEX]
        sample_cnt_paged = cb[CLIP_BUILT_PAGED_SAMPLE_COUNT_INDEX]
        unique_sample_cnt = cb[CLIP_BUILT_UNIQUE_SAMPLE_COUNT_INDEX]
        sample_cnt_resident = cb[CLIP_BUILT_RESIDENT_SAMPLE_COUNT_INDEX]
        anim_fps_resident = cb[CLIP_BUILT_RESIDENT_FPS_INDEX]
        joint_count = cb[CLIP_BUILT_JOINT_COUNT_INDEX]
        elem_count = cb[CLIP_BUILT_ELEM_COUNT_INDEX]
        stride = cb[CLIP_BUILT_STRIDE_INDEX]
        elem_large_count = cb[CLIP_BUILT_LARGE_ELEM_COUNT_INDEX]

        if looks_like_camera_animclip(data, blocks, table_end, cb):
            if not (clip_flags & FLAG_CAMERA):
                self.report({'WARNING'}, "AnimClip is missing the Camera flag but has zero joints, motion data, and camera tracks; importing it as a camera clip.")
            return import_camera_animclip(context, self, data, blocks, table_end, cb, filepath, wm)

        arm = _resolve_anim_armature(context)
        if not arm or arm.type != 'ARMATURE':
            show_popup(
                "No Armature Selected",
                "Please select an Armature in the 3D Viewport\nbefore importing this skeletal AnimClip.",
                icon='ERROR'
            )
            self.report({'ERROR'}, "Select an Armature first, then import this skeletal AnimClip.")
            return {'CANCELLED'}

        has_paged_samples = (
            BLOCK_HASHES["AnimClipSampleDataPaged"] in blocks
            and sample_cnt_paged > 0
        )
        has_resident_samples = (
            BLOCK_HASHES["AnimClipSampleDataResident"] in blocks
            and sample_cnt_resident > 0
        )

        def derive_fps(sample_count):
            if anim_duration <= 0 or sample_count <= 0:
                return 0.0
            sample_span = sample_count if (clip_flags & FLAG_LOOPING) else max(1, sample_count - 1)
            return sample_span / anim_duration

        clip_sample_cnt_full = max(
            1,
            int(sample_cnt_paged) if int(sample_cnt_paged) > 0
            else (int(sample_cnt_resident) if int(sample_cnt_resident) > 0 else 1),
        )
        clip_fps_full = anim_fps_paged if anim_fps_paged > 0 else derive_fps(clip_sample_cnt_full)
        if clip_fps_full <= 0:
            clip_fps_full = DEFAULT_IMPORT_FPS
        resident_sample_cnt_meta = max(
            1,
            int(sample_cnt_resident) if int(sample_cnt_resident) > 0 else clip_sample_cnt_full,
        )
        resident_fps_meta = anim_fps_resident if anim_fps_resident > 0 else derive_fps(resident_sample_cnt_meta)
        if resident_fps_meta <= 0:
            resident_fps_meta = clip_fps_full

        if has_paged_samples:
            sample_data_block_hash = BLOCK_HASHES["AnimClipSampleDataPaged"]
            sample_cnt = sample_cnt_paged
            anim_fps = anim_fps_paged if anim_fps_paged > 0 else derive_fps(sample_cnt)
            sample_source = "paged"
        elif has_resident_samples:
            sample_data_block_hash = BLOCK_HASHES["AnimClipSampleDataResident"]
            sample_cnt = sample_cnt_resident
            anim_fps = anim_fps_resident if anim_fps_resident > 0 else derive_fps(sample_cnt)
            sample_source = "resident"
        else:
            sample_data_block_hash = BLOCK_HASHES["AnimClipSampleDataResident"]
            sample_cnt = sample_cnt_paged if sample_cnt_paged > 0 else 1
            anim_fps = anim_fps_paged if anim_fps_paged > 0 else derive_fps(sample_cnt)
            sample_source = "header"

        if anim_fps <= 0:
            anim_fps = 30.0
        if sample_cnt <= 0:
            sample_cnt = 1
        if joint_count <= 0:
            self.report({'ERROR'}, f"AnimClipBuilt has invalid joint count: {joint_count}.")
            return {'CANCELLED'}
        if elem_count > 0 and stride <= 0:
            self.report({'ERROR'}, f"AnimClipBuilt has sample elements but invalid stride: {stride}.")
            return {'CANCELLED'}
        if stride % 4:
            self.report({'ERROR'}, f"AnimClipBuilt sample stride is not 4-byte aligned: {stride}.")
            return {'CANCELLED'}
        if BLOCK_HASHES["AnimClipBaseState"] not in blocks:
            self.report({'ERROR'}, "AnimClipBaseState block is missing.")
            return {'CANCELLED'}
        if BLOCK_HASHES["AnimClipSampleElem"] not in blocks:
            self.report({'ERROR'}, "AnimClipSampleElem block is missing.")
            return {'CANCELLED'}
        if sample_data_block_hash not in blocks:
            self.report({'ERROR'}, "AnimClip sample data block is missing.")
            return {'CANCELLED'}

        base_off, base_size = blocks[BLOCK_HASHES["AnimClipBaseState"]]
        base_size_required = joint_count * ANIM_JOINT_BASE_STATE_SIZE
        if base_size < base_size_required:
            self.report({'ERROR'}, f"AnimClipBaseState is truncated: {base_size} bytes, expected {base_size_required}.")
            return {'CANCELLED'}

        e_off, e_size = blocks[BLOCK_HASHES["AnimClipSampleElem"]]
        elem_small_size = (elem_count * U32_FIELD_BYTES) + elem_count
        elem_large_off_rel = align4(elem_small_size)
        elem_size_required = elem_large_off_rel + (elem_large_count * U32_FIELD_BYTES) + (elem_large_count * U16_FIELD_BYTES)
        if e_size < elem_size_required:
            self.report({'ERROR'}, f"AnimClipSampleElem is truncated: {e_size} bytes, expected {elem_size_required}.")
            return {'CANCELLED'}

        if BLOCK_HASHES["AnimClipJointHashes"] in blocks:
            _jh_off, jh_size = blocks[BLOCK_HASHES["AnimClipJointHashes"]]
            jh_size_required = joint_count * U32_FIELD_BYTES
            if jh_size < jh_size_required:
                self.report({'ERROR'}, f"AnimClipJointHashes is truncated: {jh_size} bytes, expected {jh_size_required}.")
                return {'CANCELLED'}
        
        self.report({'INFO'}, f"Animation: {sample_cnt} {sample_source} samples @ {anim_fps:.2f} FPS (Dur: {anim_duration:.2f}s)")
        context.scene.use_smooth_playback = False
        context.scene["engine_base_fps"] = anim_fps
        if hasattr(context.scene, "engine_anim_fps"):
            context.scene.engine_anim_fps = anim_fps
        else:
            _set_scene_fps(context.scene, anim_fps)
        context.scene.render.frame_map_old = 100
        context.scene.render.frame_map_new = 100
        if hasattr(context.scene, "engine_export_frame_start"):
            context.scene.engine_export_frame_start = 0
            context.scene.engine_export_frame_end = max(0, sample_cnt - 1)
            context.scene.engine_export_use_original_values = True







        bone_hashes = {string_crc32(b.name): b.name for b in arm.data.bones}
        joint_names = []
        missing_hashes = []
        if BLOCK_HASHES["AnimClipJointHashes"] in blocks:
            h_off, _ = blocks[BLOCK_HASHES["AnimClipJointHashes"]]
            jh_arr = struct.unpack_from(f"<{joint_count}I", data, h_off)
            for jh in jh_arr:
                name = bone_hashes.get(jh)
                if name is None:
                    name = f"Joint_{jh:08X}"
                    missing_hashes.append((jh, name))
                joint_names.append(name)
        else:
            joint_names = [f"Joint_{i}" for i in range(joint_count)]

        if missing_hashes:
            self.report({'INFO'}, f"Creating {len(missing_hashes)} placeholder bones for joints missing from model")
            prev_mode = arm.mode

            root_bone_name = None
            for b in arm.data.bones:
                if b.parent is None:
                    root_bone_name = b.name
                    break
            bpy.ops.object.mode_set(mode='EDIT')
            try:
                for jh, name in missing_hashes:
                    if name in arm.data.edit_bones:
                        continue
                    eb = arm.data.edit_bones.new(name)
                    eb.head = (0.0, 0.0, 0.0)
                    eb.tail = (0.0, 0.05, 0.0)
                    if root_bone_name and root_bone_name in arm.data.edit_bones:
                        eb.parent = arm.data.edit_bones[root_bone_name]
            finally:
                bpy.ops.object.mode_set(mode=prev_mode if prev_mode != 'EDIT' else 'OBJECT')






        full_jh_block_size = blocks[BLOCK_HASHES["AnimClipJointHashes"]][1] if BLOCK_HASHES["AnimClipJointHashes"] in blocks else 0
        full_hash_count = full_jh_block_size // U32_FIELD_BYTES
        if full_hash_count > 0:
            full_jh_arr = struct.unpack_from(f"<{full_hash_count}I", data, h_off)

            arm["engine_clip_joint_hashes"] = [
                to_signed_32(h) for h in full_jh_arr
            ]
            arm["engine_clip_joint_count"] = joint_count

        clip_flags_raw = cb[CLIP_BUILT_FLAGS_INDEX]
        arm["engine_clip_flags"] = to_signed_32(clip_flags_raw)
        set_original_flags(arm, clip_flags_raw)
        clip_name_hash_raw = cb[CLIP_BUILT_NAME_HASH_INDEX]
        arm["engine_clip_name_hash"] = to_signed_32(clip_name_hash_raw)
        for prop_name, prop_value in (
            ("engine_clip_cycle_count", cb[CLIP_BUILT_CYCLE_COUNT_INDEX]),
            ("engine_clip_facial_name_hash", cb[CLIP_BUILT_FACIAL_NAME_HASH_INDEX]),
            ("engine_clip_sound_event_hash", cb[CLIP_BUILT_SOUND_EVENT_HASH_INDEX]),
            ("engine_clip_particle_count_max", cb[CLIP_BUILT_PARTICLE_COUNT_MAX_INDEX]),
        ):
            arm[prop_name] = to_signed_32(prop_value)

        sync_scene_flag_toggles_from_flags(context.scene, clip_flags_raw)

        bone_to_idx = {name: i for i, name in enumerate(joint_names)}
        parent_map = [-1] * joint_count
        for i, name in enumerate(joint_names):
            if name in arm.pose.bones:
                p = arm.pose.bones[name].parent
                if p and p.name in bone_to_idx:
                    parent_map[i] = bone_to_idx[p.name]


        base_raw = struct.unpack_from(f"<{BASE_STATE_SHORTS_PER_JOINT * joint_count}h", data, base_off)


        base_poses_scale = [[0]*4 for _ in range(joint_count)]
        base_poses_quat = [[0]*4 for _ in range(joint_count)]
        base_poses_trans = [[0]*4 for _ in range(joint_count)]
        for i in range(joint_count):
            b = i * BASE_STATE_SHORTS_PER_JOINT
            ls_lt = struct.unpack_from("<H", data, base_off + i*16 + 14)[0]
            ls, lt = ls_lt & 0xFF, ls_lt >> 8
            base_poses_scale[i] = [1 << ls, 1 << ls, 1 << ls, 1 << ls]
            base_poses_quat[i] = [base_raw[b], base_raw[b+1], base_raw[b+2], base_raw[b+3]]
            base_poses_trans[i] = [base_raw[b+4], base_raw[b+5], base_raw[b+6], lt]

        wm.progress_update(5)


        e_offs = struct.unpack_from(f"<{elem_count}I", data, e_off)
        e_strs = data[e_off + elem_count * SAMPLE_ELEM_OFFSET_STRIDE : e_off + elem_count * SAMPLE_ELEM_OFFSET_STRIDE + elem_count]
        le_off = align4(e_off + elem_count * SAMPLE_ELEM_OFFSET_STRIDE + elem_count)
        le_offs = struct.unpack_from(f"<{elem_large_count}I", data, le_off) if elem_large_count else ()
        le_strs = struct.unpack_from(f"<{elem_large_count}H", data, le_off + elem_large_count * SAMPLE_ELEM_OFFSET_STRIDE) if elem_large_count else ()


        elem_desc = []

        for i in range(elem_count):
            s = e_strs[i]
            bits = (s & 0xF) + 1
            err = s >> 4
            ci = e_offs[i] // SAMPLE_ELEM_OFFSET_STRIDE
            ji, si = ci // JOINT_PACKED_COMPONENT_COUNT, ci % JOINT_PACKED_COMPONENT_COUNT
            if ji >= joint_count or si >= JOINT_PACKED_COMPONENT_COUNT:
                self.report({'ERROR'}, f"AnimClipSampleElem[{i}] targets invalid joint/channel {ji}/{si}.")
                return {'CANCELLED'}
            elem_desc.append((ji, si, bits, err))

        large_desc = []
        for i in range(elem_large_count):
            s = le_strs[i]
            bits = (s & 0xFF) + 1
            err = s >> 8
            ci = le_offs[i] // SAMPLE_ELEM_OFFSET_STRIDE
            ji, si = ci // JOINT_PACKED_COMPONENT_COUNT, ci % JOINT_PACKED_COMPONENT_COUNT
            if ji >= joint_count or si >= JOINT_PACKED_COMPONENT_COUNT:
                self.report({'ERROR'}, f"AnimClipSampleElem large[{i}] targets invalid joint/channel {ji}/{si}.")
                return {'CANCELLED'}
            large_desc.append((ji, si, bits, err))

        packed_bits_per_frame = sum(bits for _ji, _si, bits, _err in elem_desc)
        packed_bits_per_frame += sum(bits for _ji, _si, bits, _err in large_desc)
        if packed_bits_per_frame > stride * BITS_PER_BYTE:
            self.report({'ERROR'}, f"AnimClip sample elements need {packed_bits_per_frame} bits but stride only stores {stride * BITS_PER_BYTE}.")
            return {'CANCELLED'}




        lookup_table = None
        d_off, d_size = blocks[sample_data_block_hash]
        d_block_end = d_off + d_size
        if clip_flags & FLAG_FRAME_DATA_LOOKUP:
            lookup_count = sample_cnt_paged + (1 if (clip_flags & FLAG_LOOPING) else 0)
            if lookup_count <= 0:
                lookup_count = sample_cnt
            lookup_bytes = lookup_count * 2
            if d_off + lookup_bytes > d_block_end:
                self.report({'ERROR'}, f"Frame lookup table is truncated: needs {lookup_bytes} bytes.")
                return {'CANCELLED'}
            lookup_table_full = struct.unpack_from(f"<{lookup_count}H", data, d_off)
            lookup_table = lookup_table_full[:sample_cnt]
            d_off += (lookup_bytes + SAMPLE_LOOKUP_ALIGN - 1) & ~(SAMPLE_LOOKUP_ALIGN - 1)
            if d_off > d_block_end:
                self.report({'ERROR'}, "Frame lookup table alignment exceeds the sample data block.")
                return {'CANCELLED'}


        if lookup_table:
            total_samples = max(lookup_table) + 1 if lookup_table else sample_cnt
            if unique_sample_cnt > total_samples:
                total_samples = unique_sample_cnt
        else:
            total_samples = sample_cnt
        raw_bytes_needed = total_samples * stride
        if d_off + raw_bytes_needed > d_block_end:
            available = max(0, d_block_end - d_off)
            self.report({'ERROR'}, f"AnimClip sample data is truncated: {available} bytes, expected {raw_bytes_needed}.")
            return {'CANCELLED'}

        sample_bytes = data[d_off : d_off + raw_bytes_needed]

        pad = (-len(sample_bytes)) & 3
        if pad:
            sample_bytes += b'\x00' * pad
        sample_bytes += b'\x00' * SAMPLE_TRAILING_PAD_BYTES

        words_array = np.frombuffer(sample_bytes, dtype=np.uint32)
        words_per_frame = stride // SAMPLE_WORD_BYTES
        total_words = len(words_array)

        if not arm.animation_data:
            arm.animation_data_create()
        action = bpy.data.actions.new(name=os.path.basename(filepath))
        arm.animation_data.action = action
        set_original_flags(action, clip_flags_raw)
        arm["engine_sample_cnt"] = sample_cnt
        for target in (arm, action):
            target["engine_clip_full_sample_count"] = int(clip_sample_cnt_full)
            target["engine_clip_full_fps"] = float(clip_fps_full)
            target["engine_clip_duration"] = float(anim_duration)
            target["engine_clip_resident_sample_count"] = int(resident_sample_cnt_meta)
            target["engine_clip_resident_fps"] = float(resident_fps_meta)
            target["engine_clip_import_sample_count"] = int(sample_cnt)
            target["engine_clip_import_fps"] = float(anim_fps)
            target["engine_clip_import_sample_source"] = str(sample_source)

        motion_block = blocks.get(BLOCK_HASHES["AnimClipMotionData"])
        if motion_block:
            motion_off, motion_size = motion_block
            import_motion_as_object_transform(
                arm, action, data[motion_off:motion_off + motion_size],
                0, max(0, sample_cnt - 1), report=self.report
            )
        else:
            import_motion_as_object_transform(
                arm, action, b"", 0, max(0, sample_cnt - 1), report=self.report
            )
        context.scene.frame_start, context.scene.frame_end = 0, sample_cnt - 1

        if "store_scene_luna_settings_for_target" in globals():
            store_scene_luna_settings_for_target(context.scene, arm)
        if "mark_luna_settings_target" in globals():
            mark_luna_settings_target(context.scene, arm)

        import base64
        import json
        passthrough = {}
        handled_hashes = {
            BLOCK_HASHES["AnimClipBuilt"],
            BLOCK_HASHES["AnimClipBaseState"],
            BLOCK_HASHES["AnimClipSampleElem"],
            BLOCK_HASHES["AnimClipSampleDataResident"],
            BLOCK_HASHES["AnimClipSampleDataPaged"],
            BLOCK_HASHES["AnimClipJointHashes"],
        }
        for b_hash, (b_off, b_size) in blocks.items():
            if b_hash not in handled_hashes:
                passthrough[str(b_hash)] = base64.b64encode(data[b_off:b_off+b_size]).decode('ascii')
        
        if passthrough:
            action["engine_passthrough_blocks"] = json.dumps(passthrough)


            tracks_block = blocks.get(BLOCK_HASHES["AnimClipTracksData"])
            if tracks_block:
                tb_off, tb_size = tracks_block
                if tb_size >= ANIM_CLIP_TRACKS_DATA_SIZE:
                    loc_count, trig_count, ev_size, marker_count, _tb_data = _read_tracks_counts(data[tb_off:tb_off + tb_size])
                    

                    trigger_block = blocks.get(BLOCK_HASHES["AnimClipTriggerData"])
                    if trigger_block and trig_count > 0:
                        trig_off, trig_size = trigger_block
                        trigger_joints_size = (loc_count + 1) * TRIGGER_LOCATOR_JOINT_STRIDE if loc_count else 0
                        records_off = trig_off + trigger_joints_size + ev_size
                        event_data_end = records_off
                        if records_off >= trig_off and records_off + (trig_count * ANIM_TRIGGER_RECORD_SIZE) <= trig_off + trig_size:

                            total_frames = max(1, sample_cnt - 1)
                            for i in range(trig_count):
                                name_hash, loc_hash, flags, time, ev_off_shifted, rad = struct.unpack_from("<IIHHHH", data, records_off + i * ANIM_TRIGGER_RECORD_SIZE)
                                ev_off = ev_off_shifted << 2
                                if trigger_joints_size <= ev_off and trig_off + ev_off + TRIGGER_PAYLOAD_HEADER_SIZE <= event_data_end:
                                    actor_hash, ev_hash = struct.unpack_from("<II", data, trig_off + ev_off)
                                    

                                    ev_name = KNOWN_EVENT_HASHES.get(ev_hash, f"Unknown_{ev_hash:08X}")

                                    payload, payload_meta = parse_databuffer_fields(
                                        data, trig_off + ev_off + TRIGGER_PAYLOAD_HEADER_SIZE, event_data_end,
                                        ev_name, include_meta=True
                                    )


                                    marker_name = f"Trigger_[{i}]_{ev_hash:08X}"
                                    marker = action.pose_markers.new(name=marker_name)

                                    marker.frame = int(round((time / float(NORMALIZED_TRIGGER_TIME_MAX)) * total_frames))
                                    

                                    prefix = f"marker_{i}_"
                                    action[f"{prefix}name_hash"] = to_signed_32(name_hash)
                                    action[f"{prefix}actor_hash"] = to_signed_32(actor_hash)
                                    action[f"{prefix}ev_hash"] = to_signed_32(ev_hash)
                                    action[f"{prefix}loc_hash"] = to_signed_32(loc_hash)
                                    action[f"{prefix}flags"] = flags
                                    action[f"{prefix}rad"] = rad
                                    action[f"{prefix}event_data_off"] = ev_off
                                    
                                    if payload:

                                        field_names = []
                                        for k, v in payload.items():
                                            _store_action_event_field(
                                                action, prefix, k, v, payload_meta.get(k, {})
                                            )
                                            field_names.append(k)
                                        action[f"{prefix}ddl_fields"] = ",".join(field_names)
                    

                    marker_block = blocks.get(BLOCK_HASHES["AnimClipMarkers"])
                    if marker_block and marker_count > 0:
                        mb_off, mb_size = marker_block
                        if mb_size >= marker_count * ANIM_CLIP_MARKER_SIZE:
                            for i in range(marker_count):
                                flags, start_time, duration, name_off, name_hash, mdata = struct.unpack_from("<BxHHHIf", data, mb_off + i * ANIM_CLIP_MARKER_SIZE)
                                marker_name = f"Marker_[{i}]_{name_hash:08X}"
                                marker = action.pose_markers.new(name=marker_name)
                                marker.frame = start_time

        wm.progress_update(8)





        pose_bones = [arm.pose.bones.get(n) for n in joint_names]
        rest_local_inv = [None] * joint_count
        rest_local = [None] * joint_count
        parent_rest = [None] * joint_count
        for i, pb in enumerate(pose_bones):
            if pb:
                rest_local[i] = pb.bone.matrix_local.copy()
                rest_local_inv[i] = rest_local[i].inverted()
                if pb.parent and parent_map[i] != -1:
                    parent_rest[i] = pb.parent.bone.matrix_local.inverted() @ rest_local[i]



        valid_bones = [i for i, pb in enumerate(pose_bones) if pb is not None]
        loc_arrays = {i: np.zeros(sample_cnt * LOCATION_COMPONENT_COUNT, dtype=np.float32) for i in valid_bones}
        rot_arrays = {i: np.zeros(sample_cnt * ROTATION_COMPONENT_COUNT, dtype=np.float32) for i in valid_bones}
        sca_arrays = {i: np.zeros(sample_cnt * LOCATION_COMPONENT_COUNT, dtype=np.float32) for i in valid_bones}


        Matrix = mathutils.Matrix
        Vector = mathutils.Vector
        Quaternion = mathutils.Quaternion
        Translation = Matrix.Translation
        Diagonal = Matrix.Diagonal
        SWZ = SWIZZLE_MAT


        progress_step = max(1, sample_cnt // 50)
        for f in range(sample_cnt):
            if f % progress_step == 0:
                wm.progress_update(8 + int(60 * (f / sample_cnt)))

            s_idx = lookup_table[f] if lookup_table else f
            base_word = s_idx * words_per_frame



            curr_scale = [list(v) for v in base_poses_scale]
            curr_quat = [list(v) for v in base_poses_quat]
            curr_trans = [list(v) for v in base_poses_trans]

            bs = FastBitStream(words_array, base_word, words_per_frame + 2, total_words)


            read_bits = bs.read_bits
            for ji, si, bits, err in elem_desc:
                delta = read_bits(bits) << err
                if ji < joint_count:
                    if si < 4:
                        curr_scale[ji][si] += delta
                    elif si < 8:
                        curr_quat[ji][si - 4] += delta
                    else:
                        curr_trans[ji][si - 8] += delta
            for ji, si, bits, err in large_desc:
                delta = read_bits(bits) << err
                if ji < joint_count:
                    if si < 4:
                        curr_scale[ji][si] += delta
                    elif si < 8:
                        curr_quat[ji][si - 4] += delta
                    else:
                        curr_trans[ji][si - 8] += delta



            frame_global = [None] * joint_count
            for i in range(joint_count):
                cs = curr_scale[i]
                cq = curr_quat[i]
                ct = curr_trans[i]
                t_div = float(1 << ct[3])
                s_div = float(cs[3]) if cs[3] != 0 else 1.0
                loc = Translation((ct[0] / t_div, ct[1] / t_div, ct[2] / t_div))
                q = Quaternion((cq[3], cq[0], cq[1], cq[2]))
                q.normalize()
                qm = q.to_matrix().to_4x4()
                sm = Diagonal((cs[0] / s_div, cs[1] / s_div, cs[2] / s_div, 1.0))
                local_m = loc @ qm @ sm
                p = parent_map[i]
                if p == -1:
                    frame_global[i] = SWZ @ local_m
                else:
                    pm = frame_global[p]

                    if pm is None:

                        stack = [i]
                        cur = p
                        while cur != -1 and frame_global[cur] is None:
                            stack.append(cur)
                            cur = parent_map[cur]

                        for k in reversed(stack):
                            kp = parent_map[k]
                            kcs, kcq, kct = curr_scale[k], curr_quat[k], curr_trans[k]
                            kt_div = float(1 << kct[3])
                            ks_div = float(kcs[3]) if kcs[3] != 0 else 1.0
                            kloc = Translation((kct[0]/kt_div, kct[1]/kt_div, kct[2]/kt_div))
                            kq = Quaternion((kcq[3], kcq[0], kcq[1], kcq[2])); kq.normalize()
                            klm = kloc @ kq.to_matrix().to_4x4() @ Diagonal((kcs[0]/ks_div, kcs[1]/ks_div, kcs[2]/ks_div, 1.0))
                            if kp == -1:
                                frame_global[k] = SWZ @ klm
                            else:
                                frame_global[k] = frame_global[kp] @ klm
                    else:
                        frame_global[i] = pm @ local_m


            for i in valid_bones:
                target_m = frame_global[i]
                p = parent_map[i]
                if p != -1 and parent_rest[i] is not None:
                    parent_m = frame_global[p]
                    basis = (parent_m @ parent_rest[i]).inverted() @ target_m
                else:
                    basis = rest_local_inv[i] @ target_m

                l, r, s = basis.decompose()
                idx3 = f * 3
                idx4 = f * ROTATION_COMPONENT_COUNT
                la = loc_arrays[i]
                la[idx3] = l.x; la[idx3+1] = l.y; la[idx3+2] = l.z
                ra = rot_arrays[i]
                ra[idx4] = r.w; ra[idx4+1] = r.x; ra[idx4+2] = r.y; ra[idx4+3] = r.z
                sa = sca_arrays[i]
                sa[idx3] = s.x; sa[idx3+1] = s.y; sa[idx3+2] = s.z

        wm.progress_update(70)


        frame_indices = np.arange(sample_cnt, dtype=np.float32)


        def get_or_create_fc(target_path, target_index):
            if hasattr(action, "fcurve_ensure_for_datablock"):
                return action.fcurve_ensure_for_datablock(datablock=arm, data_path=target_path, index=target_index)
            return action.fcurves.new(data_path=target_path, index=target_index)

        for bi, i in enumerate(valid_bones):
            if bi % 16 == 0:
                wm.progress_update(70 + int(28 * (bi / max(1, len(valid_bones)))))
                
            bn = joint_names[i]
            bn_escaped = bn.replace('"', '\\"')
            path_loc = f'pose.bones["{bn_escaped}"].location'
            path_rot = f'pose.bones["{bn_escaped}"].rotation_quaternion'
            path_sca = f'pose.bones["{bn_escaped}"].scale'

            la, ra, sa = loc_arrays[i], rot_arrays[i], sca_arrays[i]


            for d in range(3):
                fc = get_or_create_fc(path_loc, d)
                fc.keyframe_points.add(sample_cnt)
                co = np.empty(sample_cnt * 2, dtype=np.float32)
                co[0::2] = frame_indices
                co[1::2] = la[d::3]
                fc.keyframe_points.foreach_set("co", co)
                fc.update()


            for d in range(4):
                fc = get_or_create_fc(path_rot, d)
                fc.keyframe_points.add(sample_cnt)
                co = np.empty(sample_cnt * 2, dtype=np.float32)
                co[0::2] = frame_indices
                co[1::2] = ra[d::4]
                fc.keyframe_points.foreach_set("co", co)
                fc.update()


            for d in range(3):
                fc = get_or_create_fc(path_sca, d)
                fc.keyframe_points.add(sample_cnt)
                co = np.empty(sample_cnt * 2, dtype=np.float32)
                co[0::2] = frame_indices
                co[1::2] = sa[d::3]
                fc.keyframe_points.foreach_set("co", co)
                fc.update()

        wm.progress_update(100)
        self.report({'INFO'}, f"Imported {sample_cnt} frames for {len(valid_bones)} bones")
        return {'FINISHED'}
