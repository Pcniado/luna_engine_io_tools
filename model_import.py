# This module is part of the public-release split of blender_import_model_anim_release.py.
# Behavior-sensitive logic was moved mechanically; avoid algorithm changes here.

from .utils import *

MODEL_BIND_POSE_FLOATS_PER_JOINT = 12
MODEL_JOINT_RECORD_SIZE = 16
MODEL_JOINT_NAME_BYTES = 256
MODEL_SUBSET_RECORD_SIZE = 128
MODEL_SUBSET_MPU_OFFSET = 24
MODEL_SUBSET_INDEX_COUNT_OFFSET = 0
MODEL_SUBSET_VERTEX_COUNT_OFFSET = 4
MODEL_SUBSET_INDEX_DATA_OFFSET = 12
MODEL_SUBSET_FLAGS_OFFSET = 20
MODEL_SUBSET_UV_LOG_OFFSET = 22
MODEL_SUBSET_MATERIAL_INDEX_OFFSET = 28
MODEL_SUBSET_VERTEX_STD_OFFSET = 64
MODEL_SUBSET_VERTEX_UV12_OFFSET = 68
MODEL_SUBSET_SKIN_OFFSET = 76
MODEL_SUBSET_CLUSTER_OFFSET = 80
MODEL_SUBSET_BASE_OFFSET = 88
MODEL_SUBSET_LOD_MASK_OFFSET = 102
MODEL_VERTEX_SHORTS_PER_VERTEX = 8
MODEL_MATERIAL_INFO_SIZE = 16
MODEL_MATERIAL_SIZE = 16
MODEL_LOOK_SIZE = 64
MODEL_LOOK_BUILT_SIZE = 80
MODEL_LOOK_GROUP_SIZE = 24
MODEL_BUILT_BSPHERE_OFFSET = 32
MODEL_BUILT_AABB_EXTENTS_OFFSET = 48
MODEL_BUILT_COMMON_MPU_OFFSET = 60
MODEL_BUILT_VERTEX_MPU_OFFSET = 64
MODEL_UV_FLOAT_TO_FIXED_BASE = 16384.0
TRIANGLE_INDEX_COUNT = 3
SUBSET_FLAG_SKINNED = 0x0001
SUBSET_FLAG_HAS_UV1 = 0x0004
SUBSET_FLAG_HAS_UV2 = 0x0008
SKIN_CLUSTER_VERTEX_COUNT = 32
SKIN_CLUSTER_WORD_BYTES = 4
SKIN_CLUSTER_OFFSET_MASK = 0xFFFFF
SKIN_CLUSTER_FULL_INDEX_BIT = 1 << 20
SKIN_CLUSTER_INFLUENCE_SHIFT = 21
SKIN_CLUSTER_JOINT_OFFSET_SHIFT = 25
SKIN_CLUSTER_NIBBLE_MASK = 0xF
SKIN_WEIGHT_SCALE = 256.0

def _read_c_string(data, offset, max_len=4096):
    offset = int(offset)
    if offset < 0 or offset >= len(data):
        return ""
    end = data.find(b'\x00', offset, min(len(data), offset + max_len))
    if end < 0:
        end = min(len(data), offset + max_len)
    return data[offset:end].decode('ascii', errors='ignore')


def _uv_fixed_to_float(uv_log_scales, channel):
    shift = (int(uv_log_scales) >> (channel * 4)) & 0xF
    return float(1 << shift) / MODEL_UV_FLOAT_TO_FIXED_BASE


def _engine_uv_to_blender(uv):
    return (float(uv[0]), 1.0 - float(uv[1]))


def _parse_model_materials(data, blocks):
    material_block = blocks.get(BLOCK_HASHES.get("ModelMaterial"))
    if not material_block:
        return []
    mat_off, mat_size = material_block
    record_size = MODEL_MATERIAL_INFO_SIZE + MODEL_MATERIAL_SIZE
    material_count = mat_size // record_size
    materials = []
    runtime_base = mat_off + material_count * MODEL_MATERIAL_INFO_SIZE
    for index in range(material_count):
        info_ptr = mat_off + index * MODEL_MATERIAL_INFO_SIZE
        runtime_ptr = runtime_base + index * MODEL_MATERIAL_SIZE
        asset_name_offset, _pad0, mapping_name_offset, _pad1 = struct.unpack_from("<IIII", data, info_ptr)
        material_id, mapping_hash, flags = struct.unpack_from("<QII", data, runtime_ptr)
        asset_path = _read_c_string(data, asset_name_offset)
        mapping_name = _read_c_string(data, mapping_name_offset)
        materials.append({
            "index": index,
            "path": asset_path,
            "mapping": mapping_name,
            "id": material_id,
            "mapping_hash": mapping_hash,
            "flags": flags,
        })
    return materials


def _material_display_name(material_info):
    path = str(material_info.get("path", "") or "")
    if path:
        return path
    mapping = str(material_info.get("mapping", "") or "")
    if mapping:
        return mapping if mapping.lower().endswith(".material") else f"{mapping}.material"
    return f"Engine_Material_{int(material_info.get('index', 0)):03d}.material"


def _make_import_material(material_info):
    path = str(material_info.get("path", "") or "")
    if path and not path.lower().endswith(".material"):
        path = f"{path}.material"
    display_name = _material_display_name({**material_info, "path": path})
    if path:
        for existing in bpy.data.materials:
            if str(getattr(existing, "engine_material_path", "") or "") == path:
                existing.name = display_name
                mat = existing
                break
        else:
            mat = bpy.data.materials.new(display_name)
    else:
        mat = bpy.data.materials.new(display_name)
    mapping = str(material_info.get("mapping", "") or "")
    mat.engine_material_path = path
    mat.engine_material_mapping_name = mapping
    mat["engine_material_index"] = int(material_info.get("index", 0))
    mat["engine_material_id"] = f"0x{int(material_info.get('id', 0)) & U64_MASK:016X}"
    mat["engine_material_mapping_hash"] = to_signed_32(material_info.get("mapping_hash", 0))
    mat["engine_material_flags"] = to_signed_32(material_info.get("flags", 0))
    return mat


def _model_block_name_map():
    names = {}
    for name, value in BLOCK_HASHES.items():
        names[int(value) & U32_MASK] = name
    return names


def _parse_model_blocks_metadata(data, blocks):
    names = _model_block_name_map()
    result = []
    for block_hash, (offset, size) in sorted(blocks.items(), key=lambda item: (item[1][0], item[0])):
        result.append({
            "hash": int(block_hash) & U32_MASK,
            "name": names.get(int(block_hash) & U32_MASK, f"0x{int(block_hash) & U32_MASK:08X}"),
            "offset": int(offset),
            "size": int(size),
        })
    return result


def _parse_model_looks_metadata(data, blocks):
    look_block = blocks.get(BLOCK_HASHES.get("ModelLook"))
    built_block = blocks.get(BLOCK_HASHES.get("ModelLookBuilt"))
    if not look_block:
        return []
    look_off, look_size = look_block
    look_count = look_size // MODEL_LOOK_SIZE
    built_off, built_size = built_block if built_block else (0, 0)
    looks = []
    for look_index in range(look_count):
        ptr = look_off + look_index * MODEL_LOOK_SIZE
        lods = []
        for lod_index in range(8):
            start, count = struct.unpack_from("<HH", data, ptr + lod_index * 4)
            lods.append({"start": int(start), "count": int(count)})

        name = f"Look {look_index}"
        name_hash = 0
        subset_ids = []
        built_ptr = built_off + look_index * MODEL_LOOK_BUILT_SIZE
        if built_block and built_ptr + MODEL_LOOK_BUILT_SIZE <= built_off + built_size:
            subset_ids_offset = struct.unpack_from("<Q", data, built_ptr)[0]
            subset_id_count = struct.unpack_from("<H", data, built_ptr + 56)[0]
            name_hash, _name_hash_lower, name_offset = struct.unpack_from("<3I", data, built_ptr + 68)
            read_name = _read_c_string(data, name_offset)
            if read_name:
                name = read_name
            ids_ptr = built_off + int(subset_ids_offset)
            ids_end = ids_ptr + int(subset_id_count) * 2
            if built_off <= ids_ptr <= ids_end <= built_off + built_size and subset_id_count:
                subset_ids = list(struct.unpack_from(f"<{int(subset_id_count)}H", data, ids_ptr))

        looks.append({
            "index": int(look_index),
            "name": name,
            "name_hash": int(name_hash) & U32_MASK,
            "lods": lods,
            "subset_ids": [int(value) for value in subset_ids],
        })
    return looks


def _collect_model_lod_subset_ids(looks, lod_index):
    ids = set()
    for look in looks or []:
        subset_ids = look.get("subset_ids", [])
        lods = look.get("lods", [])
        if not isinstance(subset_ids, list) or not isinstance(lods, list):
            continue
        try:
            lod = lods[max(0, min(int(lod_index), len(lods) - 1))]
            start = int(lod.get("start", 0))
            count = int(lod.get("count", 0))
        except Exception:
            continue
        if start < 0 or count <= 0:
            continue
        for value in subset_ids[start:start + count]:
            try:
                ids.add(int(value))
            except Exception:
                pass
    return ids


def _parse_model_look_groups_metadata(data, blocks):
    group_block = blocks.get(BLOCK_HASHES.get("ModelLookGroup"))
    if not group_block:
        return []
    group_off, group_size = group_block
    if group_size <= 0:
        return []
    group_count = data[group_off]
    records_base = group_off + 1
    groups = []
    for group_index in range(min(int(group_count), max(0, (group_size - 1) // MODEL_LOOK_GROUP_SIZE))):
        ptr = records_base + group_index * MODEL_LOOK_GROUP_SIZE
        if ptr + MODEL_LOOK_GROUP_SIZE > group_off + group_size:
            break
        indices_offset, look_count = struct.unpack_from("<QH", data, ptr)
        name_hash, name_offset = struct.unpack_from("<II", data, ptr + 16)
        name = _read_c_string(data, name_offset) or f"Look Group {group_index}"
        indices_ptr = records_base + int(indices_offset)
        indices_end = indices_ptr + int(look_count) * 2
        look_indices = []
        if records_base <= indices_ptr <= indices_end <= group_off + group_size and look_count:
            look_indices = list(struct.unpack_from(f"<{int(look_count)}H", data, indices_ptr))
        groups.append({
            "index": int(group_index),
            "name": name,
            "name_hash": int(name_hash) & U32_MASK,
            "look_indices": [int(value) for value in look_indices],
        })
    return groups


def _store_model_metadata(arm, filepath, data, blocks, sb_offset, source_had_stg, dat1_version, joint_count, material_count):
    block_count = struct.unpack_from("<H", data, 12)[0] if len(data) >= 14 else len(blocks)
    fixup_count = struct.unpack_from("<H", data, 14)[0] if len(data) >= 16 else 0
    geom_block = blocks.get(BLOCK_HASHES.get("ModelSubsetGeomData"), (0, 0))
    subset_block = blocks.get(BLOCK_HASHES.get("ModelSubset"), (0, 0))
    first_block_offset = min((offset for offset, size in blocks.values() if size or offset), default=sb_offset)

    arm["engine_mpu"] = float(arm.get("engine_mpu", 1.0))
    arm["engine_model_joint_count"] = int(joint_count)
    arm["engine_model_source_path"] = os.path.abspath(filepath)
    arm["engine_model_source_had_stg"] = bool(source_had_stg)
    arm["engine_model_dat1_version"] = to_signed_32(dat1_version)
    arm["engine_model_dat1_size"] = int(len(data))
    arm["engine_model_block_count"] = int(block_count)
    arm["engine_model_fixup_count"] = int(fixup_count)
    arm["engine_model_string_table_size"] = max(0, int(first_block_offset) - int(sb_offset))
    arm["engine_model_material_count"] = int(material_count)
    arm["engine_model_subset_count"] = int(subset_block[1] // MODEL_SUBSET_RECORD_SIZE) if subset_block else 0
    arm["engine_model_geom_offset"] = int(geom_block[0])
    arm["engine_model_geom_size"] = int(geom_block[1])
    arm["engine_model_blocks_json"] = json.dumps(_parse_model_blocks_metadata(data, blocks), separators=(",", ":"))
    arm["engine_model_looks_json"] = json.dumps(_parse_model_looks_metadata(data, blocks), separators=(",", ":"))
    arm["engine_model_look_groups_json"] = json.dumps(_parse_model_look_groups_metadata(data, blocks), separators=(",", ":"))

    model_built = blocks.get(BLOCK_HASHES.get("ModelBuilt"))
    if model_built:
        mb_offset, mb_size = model_built
        try:
            if mb_size >= MODEL_BUILT_VERTEX_MPU_OFFSET + 4:
                bsphere = struct.unpack_from("<4f", data, mb_offset + MODEL_BUILT_BSPHERE_OFFSET)
                aabb = struct.unpack_from("<3f", data, mb_offset + MODEL_BUILT_AABB_EXTENTS_OFFSET)
                common_mpu = struct.unpack_from("<f", data, mb_offset + MODEL_BUILT_COMMON_MPU_OFFSET)[0]
                vertex_mpu = struct.unpack_from("<f", data, mb_offset + MODEL_BUILT_VERTEX_MPU_OFFSET)[0]
                if (
                    all(math.isfinite(v) for v in bsphere + aabb)
                    and bsphere[3] > 0.0
                    and all(v >= 0.0 for v in aabb)
                ):
                    arm["engine_model_source_bsphere_json"] = json.dumps(list(bsphere), separators=(",", ":"))
                    arm["engine_model_source_aabb_json"] = json.dumps(list(aabb), separators=(",", ":"))
                if math.isfinite(common_mpu) and common_mpu > 0.0:
                    arm["engine_model_source_common_mpu"] = float(common_mpu)
                if math.isfinite(vertex_mpu) and vertex_mpu > 0.0:
                    arm["engine_model_source_vertex_mpu"] = float(vertex_mpu)
                    arm["engine_mpu"] = float(vertex_mpu)
                elif math.isfinite(common_mpu) and common_mpu > 0.0:
                    arm["engine_mpu"] = float(common_mpu)
        except Exception:
            pass


def _set_mesh_uv_layer(mesh, layer_name, vertex_indices, vertex_uvs):
    uv_layer = mesh.uv_layers.new(name=layer_name)
    if vertex_uvs is None or len(vertex_uvs) == 0:
        for item in uv_layer.data:
            item.uv = _engine_uv_to_blender((0.0, 0.0))
        return
    for loop_index, vertex_index in enumerate(vertex_indices):
        uv = vertex_uvs[int(vertex_index)]
        uv_layer.data[loop_index].uv = _engine_uv_to_blender(uv)


class ImportEngineModel(Operator, ImportHelper):
    bl_idname = "import_scene.engine_model"
    bl_label = "Import Luna Engine Model"
    bl_description = "Import one or more engine-compiled binary Model (.model) files"
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}
    filename_ext = ".model"
    filter_glob: StringProperty(default="*.model;*.dat1", options={'HIDDEN'})
    files: CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})
    import_all_lods: BoolProperty(
        name="Import All LODs",
        description="Import lower-detail subsets too; off imports only LOD0 subsets",
        default=False,
        options={'SKIP_SAVE'},
    )

    def draw(self, context):
        self.layout.prop(self, "import_all_lods")

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
            self.report({'ERROR'}, "No file selected. Pick at least one .model file to import.")
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
                    log_exception("Unexpected error importing model %s", filepath)
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
            detail = "; ".join(failure_messages) if failure_messages else "no files imported"
            self.report({'ERROR'}, f"Model import failed: {detail}")
            return {'CANCELLED'}

        if failure_messages:
            self.report({'WARNING'}, f"Imported {success_count}/{len(filepaths)} models. Skipped: {'; '.join(failure_messages)}")
        else:
            self.report({'INFO'}, f"Imported {success_count} model{'s' if success_count != 1 else ''}.")
        return {'FINISHED'}

    def _do_import(self, context, wm, filepath):
        data, blocks, sb_offset = get_dat1_data(filepath)
        if not data:
            self.report({'ERROR'}, f"{os.path.basename(filepath)}: invalid DAT1 magic signature not found.")
            return {'CANCELLED'}
        required_model_blocks = ("ModelBuilt", "ModelMaterial", "ModelLook", "ModelLookBuilt", "ModelLookGroup", "ModelSubset", "ModelSubsetGeomData")
        missing_model_blocks = [name for name in required_model_blocks if BLOCK_HASHES[name] not in blocks]
        if missing_model_blocks:
            self.report({'ERROR'}, f"{os.path.basename(filepath)}: file does not contain Model data ({', '.join(missing_model_blocks)} missing).")
            return {'CANCELLED'}

        wm.progress_update(5)
        import_all_lods = bool(getattr(self, "import_all_lods", False))
        lod0_subset_ids = _collect_model_lod_subset_ids(_parse_model_looks_metadata(data, blocks), 0)
        has_skeleton = all(
            BLOCK_HASHES[name] in blocks
            for name in ("ModelJointHierarchy", "ModelJoint", "ModelBindPose")
        )

        dat1_version = struct.unpack_from("<I", data, 4)[0]
        source_had_stg = False
        try:
            with open(filepath, "rb") as src:
                source_had_stg = src.read(4) == b"STG\x00"
        except OSError:
            pass

        mpu = 1.0
        if BLOCK_HASHES["ModelSubset"] in blocks:
            s_off, _ = blocks[BLOCK_HASHES["ModelSubset"]]
            mpu = struct.unpack_from("<f", data, s_off + MODEL_SUBSET_MPU_OFFSET)[0]
        material_infos = _parse_model_materials(data, blocks)
        imported_materials = [_make_import_material(material_info) for material_info in material_infos]

        joint_count = 0
        joints = []
        if has_skeleton:
            h_off, _ = blocks[BLOCK_HASHES["ModelJointHierarchy"]]
            joint_count = struct.unpack_from("<HHHH", data, h_off)[1]
            j_off, _ = blocks[BLOCK_HASHES["ModelJoint"]]
            b_off, _ = blocks[BLOCK_HASHES["ModelBindPose"]]

            bp_raw = struct.unpack_from(f"<{MODEL_BIND_POSE_FLOATS_PER_JOINT * joint_count}f", data, b_off)
            local_bind_mats = []
            for i in range(joint_count):
                base = i * MODEL_BIND_POSE_FLOATS_PER_JOINT
                loc = mathutils.Matrix.Translation((bp_raw[base+8], bp_raw[base+9], bp_raw[base+10]))
                rot = mathutils.Quaternion((bp_raw[base+7], bp_raw[base+4], bp_raw[base+5], bp_raw[base+6])).to_matrix().to_4x4()
                sca = mathutils.Matrix.Diagonal((bp_raw[base], bp_raw[base+1], bp_raw[base+2], 1.0))
                local_bind_mats.append(loc @ rot @ sca)

            j_raws = list(struct.iter_unpack("<hHHHII", data[j_off:j_off + joint_count * MODEL_JOINT_RECORD_SIZE]))
            for i in range(joint_count):
                j_raw = j_raws[i]
                name_addr = j_off + i * MODEL_JOINT_RECORD_SIZE + j_raw[5]
                name = data[name_addr:name_addr + MODEL_JOINT_NAME_BYTES].split(b'\x00', 1)[0].decode('ascii', errors='ignore')
                if not name or not all(c.isprintable() for c in name):
                    sb_name_addr = sb_offset + j_raw[5]
                    name = data[sb_name_addr:sb_name_addr + MODEL_JOINT_NAME_BYTES].split(b'\x00', 1)[0].decode('ascii', errors='ignore')
                if not name:
                    name = f"Joint_{j_raw[4]:08X}"
                joints.append({'parent': j_raw[0], 'name': name, 'local_mat': local_bind_mats[i]})

        wm.progress_update(15)


        bpy.ops.object.armature_add(enter_editmode=True)
        arm = context.active_object
        arm.name = os.path.basename(filepath)
        while arm.data.edit_bones:
            arm.data.edit_bones.remove(arm.data.edit_bones[0])

        global_mats = [None] * joint_count
        if has_skeleton:
            for i in range(joint_count):
                eb = arm.data.edit_bones.new(joints[i]['name'])
                joints[i]['blender_name'] = eb.name

            for i, j in enumerate(joints):
                if j['parent'] != -1:
                    arm.data.edit_bones[j['blender_name']].parent = arm.data.edit_bones[joints[j['parent']]['blender_name']]

            for i in range(joint_count):
                j = joints[i]
                eb = arm.data.edit_bones[j['blender_name']]
                if j['parent'] == -1:
                    global_mats[i] = SWIZZLE_MAT @ j['local_mat']
                else:
                    global_mats[i] = global_mats[j['parent']] @ j['local_mat']
                eb.matrix = global_mats[i]
                eb.tail = eb.head + (global_mats[i].to_3x3() @ mathutils.Vector((0, 0.1, 0)))

        bpy.ops.object.mode_set(mode='OBJECT')
        for i, joint in enumerate(joints):
            arm.data.bones[joint['blender_name']]["engine_joint_index"] = i
        arm["engine_mpu"] = mpu
        arm["engine_model_static"] = not has_skeleton
        _store_model_metadata(
            arm,
            filepath,
            data,
            blocks,
            sb_offset,
            source_had_stg,
            dat1_version,
            joint_count,
            len(imported_materials),
        )
        arm["engine_model_import_all_lods"] = bool(import_all_lods)
        arm["engine_model_import_mode"] = "ALL_LODS" if import_all_lods else "LOD0"
        arm["engine_model_imported_subset_count"] = 0

        wm.progress_update(30)

        if BLOCK_HASHES["ModelSubset"] in blocks and BLOCK_HASHES["ModelSubsetGeomData"] in blocks:
            s_off, s_size = blocks[BLOCK_HASHES["ModelSubset"]]
            g_off, _ = blocks[BLOCK_HASHES["ModelSubsetGeomData"]]
            subset_count = s_size // MODEL_SUBSET_RECORD_SIZE
            joint_names_list = [j['blender_name'] for j in joints]
            imported_subset_count = 0

            for s in range(subset_count):
                wm.progress_update(30 + int(65 * (s / max(1, subset_count))))
                sub_ptr = s_off + s * MODEL_SUBSET_RECORD_SIZE
                idx_count = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_INDEX_COUNT_OFFSET)[0]
                vtx_count = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_VERTEX_COUNT_OFFSET)[0]
                idx_data_off = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_INDEX_DATA_OFFSET)[0]
                subset_flags = struct.unpack_from("<H", data, sub_ptr + MODEL_SUBSET_FLAGS_OFFSET)[0]
                subset_mpu = struct.unpack_from("<f", data, sub_ptr + MODEL_SUBSET_MPU_OFFSET)[0]
                if not math.isfinite(subset_mpu) or subset_mpu <= 0.0:
                    subset_mpu = mpu
                uv_log_scales = struct.unpack_from("<H", data, sub_ptr + MODEL_SUBSET_UV_LOG_OFFSET)[0]
                material_index = struct.unpack_from("<H", data, sub_ptr + MODEL_SUBSET_MATERIAL_INDEX_OFFSET)[0]
                v_std = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_VERTEX_STD_OFFSET)[0]
                v_uv12 = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_VERTEX_UV12_OFFSET)[0]
                s_off_v = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_SKIN_OFFSET)[0]
                c_off_v = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_CLUSTER_OFFSET)[0]
                b_off_v = struct.unpack_from("<I", data, sub_ptr + MODEL_SUBSET_BASE_OFFSET)[0]
                lod_mask = struct.unpack_from("<H", data, sub_ptr + MODEL_SUBSET_LOD_MASK_OFFSET)[0]
                if not import_all_lods:
                    is_lod0 = bool(lod_mask & 1)
                    if lod0_subset_ids:
                        is_lod0 = is_lod0 or (s in lod0_subset_ids)
                    elif lod_mask == 0:
                        is_lod0 = True
                    if not is_lod0:
                        continue

                if vtx_count == 0 or idx_count == 0:
                    continue

                v_base = g_off + b_off_v + v_std
                i_base = g_off + b_off_v + idx_data_off



                vtx_block = np.frombuffer(data, dtype=np.int16, count=vtx_count * MODEL_VERTEX_SHORTS_PER_VERTEX, offset=v_base).reshape(-1, MODEL_VERTEX_SHORTS_PER_VERTEX)
                xyz = vtx_block[:, :3].astype(np.float32) * subset_mpu
                if subset_flags & SUBSET_FLAG_ORIGIN_OFFSET:
                    xyz += _decode_subset_origin_meters(data, sub_ptr, subset_mpu)

                uv0 = vtx_block[:, 6:8].astype(np.float32) * _uv_fixed_to_float(uv_log_scales, 0)
                uv1 = None
                uv2 = None
                uv12_channel_count = int(bool(subset_flags & SUBSET_FLAG_HAS_UV1)) + int(bool(subset_flags & SUBSET_FLAG_HAS_UV2))
                if v_uv12 > 0 and uv12_channel_count:
                    uv12_base = g_off + b_off_v + v_uv12
                    uv12_values = np.frombuffer(
                        data,
                        dtype=np.int16,
                        count=vtx_count * uv12_channel_count * 2,
                        offset=uv12_base,
                    ).reshape(vtx_count, uv12_channel_count, 2).astype(np.float32)
                    if subset_flags & SUBSET_FLAG_HAS_UV1:
                        uv1 = uv12_values[:, 0, :] * _uv_fixed_to_float(uv_log_scales, 1)
                    if subset_flags & SUBSET_FLAG_HAS_UV2:
                        uv2_index = 1 if (subset_flags & SUBSET_FLAG_HAS_UV1) else 0
                        uv2 = uv12_values[:, uv2_index, :] * _uv_fixed_to_float(uv_log_scales, 2)

                verts_flat = np.empty(vtx_count * 3, dtype=np.float32)
                verts_flat[0::3] = xyz[:, 0]
                verts_flat[1::3] = -xyz[:, 2]
                verts_flat[2::3] = xyz[:, 1]


                idx_arr = np.frombuffer(data, dtype=np.uint16, count=idx_count, offset=i_base).astype(np.int32)
                tri_count = idx_count // TRIANGLE_INDEX_COUNT


                me = bpy.data.meshes.new(f"Subset_{s}")
                me.vertices.add(vtx_count)
                me.vertices.foreach_set("co", verts_flat)
                me.loops.add(tri_count * TRIANGLE_INDEX_COUNT)
                me.loops.foreach_set("vertex_index", idx_arr[:tri_count * TRIANGLE_INDEX_COUNT])
                me.polygons.add(tri_count)
                loop_starts = np.arange(0, tri_count * TRIANGLE_INDEX_COUNT, TRIANGLE_INDEX_COUNT, dtype=np.int32)
                loop_totals = np.full(tri_count, TRIANGLE_INDEX_COUNT, dtype=np.int32)
                me.polygons.foreach_set("loop_start", loop_starts)
                me.polygons.foreach_set("loop_total", loop_totals)
                loop_vertex_indices = idx_arr[:tri_count * TRIANGLE_INDEX_COUNT]
                _set_mesh_uv_layer(me, "UV0", loop_vertex_indices, uv0)
                _set_mesh_uv_layer(me, "UV1", loop_vertex_indices, uv1)
                _set_mesh_uv_layer(me, "UV2", loop_vertex_indices, uv2)
                me.update(calc_edges=True)
                me.validate(verbose=False)

                obj = bpy.data.objects.new(f"Subset_{s}", me)
                context.collection.objects.link(obj)
                if 0 <= material_index < len(imported_materials):
                    me.materials.append(imported_materials[material_index])
                    for poly in me.polygons:
                        poly.material_index = 0
                obj["engine_subset_index"] = s
                obj["engine_subset_flags"] = int(subset_flags)
                obj["engine_material_index"] = int(material_index)
                obj["engine_uv_log_scales"] = int(uv_log_scales)
                obj["engine_mpu"] = float(subset_mpu)
                obj["engine_lod_mask"] = lod_mask
                obj["engine_uv0_present"] = True
                obj["engine_uv1_present"] = bool(subset_flags & SUBSET_FLAG_HAS_UV1)
                obj["engine_uv2_present"] = bool(subset_flags & SUBSET_FLAG_HAS_UV2)
                obj.parent = arm
                if has_skeleton:
                    mod = obj.modifiers.new(type='ARMATURE', name="Armature")
                    mod.object = arm


                if has_skeleton and joint_count > 0 and s_off_v > 0 and c_off_v > 0:
                    vgs = [obj.vertex_groups.new(name=n) for n in joint_names_list]
                    c_b = g_off + b_off_v + c_off_v
                    s_b = g_off + b_off_v + s_off_v



                    group_weights = {}

                    cluster_count = (vtx_count + SKIN_CLUSTER_VERTEX_COUNT - 1) // SKIN_CLUSTER_VERTEX_COUNT

                    cluster_hdrs = struct.unpack_from(f"<{cluster_count}I", data, c_b)

                    for ic in range(cluster_count):
                        cl = cluster_hdrs[ic]
                        c_o = cl & SKIN_CLUSTER_OFFSET_MASK
                        c_inf = ((cl >> SKIN_CLUSTER_INFLUENCE_SHIFT) & SKIN_CLUSTER_NIBBLE_MASK) + 1
                        c_jo = (cl >> SKIN_CLUSTER_JOINT_OFFSET_SHIFT) & SKIN_CLUSTER_NIBBLE_MASK
                        cl_ptr = s_b + c_o * SKIN_CLUSTER_WORD_BYTES
                        is_full_idx = bool(cl & SKIN_CLUSTER_FULL_INDEX_BIT)
                        bpj = (2 if is_full_idx else 1) + (1 if c_inf > 1 else 0)

                        base_v = ic * SKIN_CLUSTER_VERTEX_COUNT
                        remaining = min(SKIN_CLUSTER_VERTEX_COUNT, vtx_count - base_v)

                        for vic in range(remaining):
                            v_idx = base_v + vic
                            v_ptr = cl_ptr + vic * (c_inf * bpj)
                            cur_ji = int(c_jo * SKIN_WEIGHT_SCALE)
                            for infl in range(c_inf):
                                jp = v_ptr + infl * bpj
                                if is_full_idx:
                                    ji = struct.unpack_from("<H", data, jp)[0]
                                    w = data[jp + 2] if c_inf > 1 else int(SKIN_WEIGHT_SCALE)
                                else:
                                    cur_ji += data[jp]
                                    ji = cur_ji
                                    w = data[jp + 1] if c_inf > 1 else int(SKIN_WEIGHT_SCALE)
                                if w > 0 and ji < joint_count:
                                    gw = group_weights.setdefault(ji, {})
                                    gw.setdefault(w, []).append(v_idx)


                    for ji, wdict in group_weights.items():
                        vg = vgs[ji]
                        for w, vlist in wdict.items():
                            vg.add(vlist, w / SKIN_WEIGHT_SCALE, 'REPLACE')
                imported_subset_count += 1
            arm["engine_model_imported_subset_count"] = int(imported_subset_count)

        arm.active_lod = 0
        update_lod_visibility(arm, context)
        wm.progress_update(100)
        mode_label = "all LODs" if import_all_lods else "LOD0"
        imported_count = int(arm.get("engine_model_imported_subset_count", 0))
        self.report({'INFO'}, f"Imported model with {joint_count} joints, {imported_count} {mode_label} subset(s)")
        return {'FINISHED'}
