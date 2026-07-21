from .utils import *
from .constants import SUBSET_CENTER_LOG_SCALE
from .dat1 import DAT1_BLOCK_TABLE_ENTRY_SIZE, DAT1_FILE_ID, DAT1_FIXUP_TABLE_ENTRY_SIZE, DAT1_HEADER_SIZE
from .hashes import BLOCK_HASHES, string_crc32
from .model_morph import (
    MORPH_DELTA_PRECISION,
    MORPH_VERTEX_DELTA_EPSILON,
    decode_model_morph2,
    encode_model_morph2,
    encode_model_smooth2,
)
from .model_import import (
    MODEL_MATERIAL_INFO_SIZE,
    MODEL_MATERIAL_SIZE,
    MODEL_SUBSET_BASE_OFFSET,
    MODEL_SUBSET_FLAGS_OFFSET,
    MODEL_SUBSET_INDEX_COUNT_OFFSET,
    MODEL_SUBSET_INDEX_DATA_OFFSET,
    MODEL_SUBSET_MATERIAL_INDEX_OFFSET,
    MODEL_SUBSET_MPU_OFFSET,
    MODEL_SUBSET_RECORD_SIZE,
    MODEL_SUBSET_UV_LOG_OFFSET,
    MODEL_SUBSET_VERTEX_COUNT_OFFSET,
    MODEL_SUBSET_VERTEX_STD_OFFSET,
    MODEL_SUBSET_VERTEX_UV12_OFFSET,
    MODEL_UV_FLOAT_TO_FIXED_BASE,
    SKIN_CLUSTER_FULL_INDEX_BIT,
    SKIN_CLUSTER_INFLUENCE_SHIFT,
    SKIN_CLUSTER_JOINT_OFFSET_SHIFT,
    SKIN_CLUSTER_OFFSET_MASK,
    SKIN_CLUSTER_VERTEX_COUNT,
    SKIN_CLUSTER_WORD_BYTES,
    SKIN_WEIGHT_SCALE,
    SUBSET_FLAG_HAS_UV1,
    SUBSET_FLAG_HAS_UV2,
    SUBSET_FLAG_SKINNED,
    _decode_packed_normal,
    _decode_packed_tangent,
    _parse_model_materials,
    _read_c_string,
)

DAT1_BLOCK_ALIGN = 16
DAT1_CACHELINE_ALIGN = 64
STG_MAGIC = 0x00475453
STG_VERSION = 0x1
STG_HEADER_ALIGN = 16

MODEL_BUILT_SIZE = 96
MODEL_LOOK_SIZE = 64
MODEL_LOOK_BUILT_SIZE = 80
MODEL_LOOK_GROUP_SIZE = 24
MODEL_STD_VERTEX_SIZE = 16
MODEL_MAX_VERTEX_COUNT = 0xFFFF
MODEL_SPLIT_VERTEX_TARGET = 60000
MODEL_DEFAULT_MPU = 1.0 / 32768.0

SUBSET_FLAG_HAS_ANIM_VERT = 0x0002
SUBSET_FLAG_HAS_COLOR = 0x0010
SUBSET_FLAG_HAS_ORIGIN_OFFSET = 0x4000
SUBSET_EXPORT_FLAG_CLEAR_MASK = (
    SUBSET_FLAG_SKINNED
    | SUBSET_FLAG_HAS_ANIM_VERT
    | SUBSET_FLAG_HAS_UV1
    | SUBSET_FLAG_HAS_UV2
    | SUBSET_FLAG_HAS_COLOR
    | SUBSET_FLAG_HAS_ORIGIN_OFFSET
)

MODEL_FLAG_HAS_SKINNING = 1 << 1
MODEL_FLAG_HAS_GPU_SKINNING = 1 << 2
MODEL_FLAG_ANIM_VERT = 1 << 28
MODEL_FLAG_ANIM_DYNAMICS = 1 << 29
MODEL_FLAG_USES_AUTO_LODS = 1 << 30

SKIN_JOINT_OFFSET_STEP = 256
SKIN_JOINT_OFFSET_MAX = SKIN_JOINT_OFFSET_STEP * 15
SKIN_UINT8_MAX = 255
SKIN_CLUSTER_ANIM_VERT_BIT = 1 << 29

MODEL_BUILT_FLAGS_OFFSET = 0
MODEL_BUILT_FADE_OUT_DIST_OFFSET = 24
MODEL_BUILT_BSPHERE_OFFSET = 32
MODEL_BUILT_AABB_EXTENTS_OFFSET = 48
MODEL_BUILT_COMMON_MPU_OFFSET = 60
MODEL_BUILT_VERTEX_MPU_OFFSET = 64
MODEL_BUILT_CUSTOM_STREAM_COUNT_OFFSET = 68
MODEL_BUILT_CONTENT_FLAGS_OFFSET = 72
MODEL_BUILT_SUBSET_LOD_MASK_COUNT_OFFSET = 74
MODEL_BUILT_STRAND_SUBSET_COUNT_OFFSET = 78

CONTENT_FLAG_ANIM_MORPH = 0x0001
CONTENT_FLAG_ANIM_ZIVA = 0x0004
CONTENT_FLAG_ANIM_VERT_SMOOTH = 0x0010
CONTENT_FLAG_USES_AUTO_LODS = 0x0080

MODEL_SUBSET_SURFACE_AREA_OFFSET = 32
MODEL_SUBSET_UV_AREA_OFFSET = 36
MODEL_SUBSET_FADE_OUT_DIST_OFFSET = 40
MODEL_SUBSET_MATERIAL_LOD_DIST_OFFSET = 44
MODEL_SUBSET_OBJ_CENTER_OFFSET = 48
MODEL_SUBSET_OBJ_EXTENTS_OFFSET = 60
MODEL_SUBSET_LONGEST_EDGE_OFFSET = 118
MODEL_SUBSET_CURVATURE_RADIUS_OFFSET = 120
MODEL_GLOBAL_BOUNDS_PADDING = 0.05

ASSET_CHUNK_UNCOMPRESSED_MASK = (1 << 30) - 1
ASSET_CHUNK_COMPRESSED_SHIFT = 30
ASSET_CHUNK_COMPRESSION_SHIFT = 60
ASSET_COMPRESSION_NONE = 0


def _align(value, alignment):
    return (int(value) + alignment - 1) & ~(alignment - 1)


def _align_buffer(buffer, alignment):
    pad = _align(len(buffer), alignment) - len(buffer)
    if pad:
        buffer += b"\x00" * pad


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _clamp_i16(value):
    return int(_clamp(int(round(value)), -32768, 32767))


def _clamp_u16(value):
    return int(_clamp(int(round(value)), 0, 65535))


def _clamp_u8(value):
    return int(_clamp(int(round(value)), 0, 255))


def _round_engine(value):
    value = float(value)
    return int(math.floor(value + 0.5)) if value >= 0.0 else int(math.ceil(value - 0.5))


def _vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _vec_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_len(a):
    return math.sqrt(max(0.0, _vec_dot(a, a)))


def _vec_normalize(a, fallback=(0.0, 0.0, 1.0)):
    length = _vec_len(a)
    if length <= 1e-8:
        return fallback
    return (a[0] / length, a[1] / length, a[2] / length)


def _luna_triangle_tangent_space(positions, uvs):
    #matches luna tangent gen
    if len(positions) != 3 or len(uvs) != 3:
        raise ValueError("Luna tangent generation requires one triangle")
    epsilon = 0.000001
    edge_01 = _vec_sub(positions[1], positions[0])
    edge_02 = _vec_sub(positions[2], positions[0])
    triangle_normal = _vec_normalize(_vec_cross(edge_01, edge_02), fallback=(0.0, 1.0, 0.0))

    min_v, mid_v, max_v = 0, 1, 2
    if uvs[max_v][1] < uvs[mid_v][1]:
        max_v, mid_v = mid_v, max_v
    if uvs[max_v][1] < uvs[min_v][1]:
        max_v, min_v = min_v, max_v
    if uvs[mid_v][1] < uvs[min_v][1]:
        mid_v, min_v = min_v, mid_v
    v_range = float(uvs[max_v][1]) - float(uvs[min_v][1])
    interp = (
        (float(uvs[mid_v][1]) - float(uvs[min_v][1])) / v_range
        if v_range > epsilon else 1.0
    )
    interp_pos = _vec_add(
        _vec_mul(positions[min_v], 1.0 - interp),
        _vec_mul(positions[max_v], interp),
    )
    interp_u = float(uvs[min_v][0]) * (1.0 - interp) + float(uvs[max_v][0]) * interp
    tangent = _vec_sub(interp_pos, positions[mid_v])
    if interp_u < float(uvs[mid_v][0]):
        tangent = _vec_mul(tangent, -1.0)
    tangent = _vec_normalize(tangent, fallback=(1.0, 0.0, 0.0))

    min_u, mid_u, max_u = 0, 1, 2
    if uvs[max_u][0] < uvs[mid_u][0]:
        max_u, mid_u = mid_u, max_u
    if uvs[max_u][0] < uvs[min_u][0]:
        max_u, min_u = min_u, max_u
    if uvs[mid_u][0] < uvs[min_u][0]:
        mid_u, min_u = min_u, mid_u
    u_range = float(uvs[max_u][0]) - float(uvs[min_u][0])
    interp = (
        (float(uvs[mid_u][0]) - float(uvs[min_u][0])) / u_range
        if u_range > epsilon else 1.0
    )
    interp_pos = _vec_add(
        _vec_mul(positions[min_u], 1.0 - interp),
        _vec_mul(positions[max_u], interp),
    )
    interp_v = float(uvs[min_u][1]) * (1.0 - interp) + float(uvs[max_u][1]) * interp
    binormal = _vec_sub(interp_pos, positions[mid_u])
    if interp_v < float(uvs[mid_u][1]):
        binormal = _vec_mul(binormal, -1.0)
    binormal = _vec_normalize(binormal, fallback=(0.0, 0.0, 1.0))

    computed_binormal = _vec_cross(tangent, triangle_normal)
    tangent_flip = 1.0 if _vec_dot(computed_binormal, binormal) > 0.0 else -1.0
    return tangent, tangent_flip


def _blender_to_engine_vec(value):
    return (float(value.x), float(value.z), -float(value.y))


def _safe_matrix_relative_to_armature(arm, obj):
    try:
        if arm:
            return arm.matrix_world.inverted() @ obj.matrix_world
        return obj.matrix_world.copy()
    except Exception:
        return getattr(obj, "matrix_local", mathutils.Matrix.Identity(4))


def _source_path_from_armature(arm):
    try:
        path = str(arm.get("engine_model_source_path", "") or "")
    except Exception:
        path = ""
    return path


def _resolve_model_armature(context):
    arm = _resolve_anim_armature(context)
    if arm:
        return arm
    active = getattr(context, "active_object", None)
    if active and getattr(active, "type", None) == "ARMATURE":
        return active
    if active and getattr(active, "parent", None) and active.parent.type == "ARMATURE":
        return active.parent
    return None


def _parse_u64(value, default=0):
    if value is None:
        return int(default) & U64_MASK
    if isinstance(value, int):
        return int(value) & U64_MASK
    text = str(value).strip()
    if not text:
        return int(default) & U64_MASK
    try:
        return int(text, 0) & U64_MASK
    except ValueError:
        return int(default) & U64_MASK


def _valid_material_id(index):
    return (0x8000000000000000 | ((int(index) + 1) & 0x3FFFFFFFFFFFFFFF)) & U64_MASK


def _normalize_material_asset_path(path):
    text = str(path or "").strip()
    text = re.sub(r"(?i)(\.material)\.\d{3}$", r"\1", text)
    if text and not text.lower().endswith(".material"):
        text = f"{text}.material"
    return text


class _StringPool:
    def __init__(self, base_offset, initial_bytes):
        self.base_offset = int(base_offset)
        self.buffer = bytearray(initial_bytes or b"")
        self.offsets = {}
        self._index_existing_strings()

    def _index_existing_strings(self):
        start = 0
        data = bytes(self.buffer)
        while start < len(data):
            end = data.find(b"\x00", start)
            if end < 0:
                break
            if end > start:
                try:
                    text = data[start:end].decode("ascii")
                except UnicodeDecodeError:
                    text = ""
                if text and text not in self.offsets:
                    self.offsets[text] = self.base_offset + start
            start = end + 1

    def add(self, text):
        text = str(text or "")
        if text in self.offsets:
            return self.offsets[text]
        encoded = text.encode("ascii", errors="ignore") + b"\x00"
        offset = self.base_offset + len(self.buffer)
        self.buffer += encoded
        self.offsets[text] = offset
        return offset

    def bytes(self):
        return bytes(self.buffer)


class _Dat1Template:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, "rb") as f:
            raw = f.read()
        dat1_offset = raw.find(b"1TAD")
        if dat1_offset < 0:
            raise ValueError("DAT1 magic not found")
        self.had_stg = raw[:4] == struct.pack("<I", STG_MAGIC)
        self.prefix = raw[:dat1_offset]
        self.data = raw[dat1_offset:]
        if len(self.data) < DAT1_HEADER_SIZE:
            raise ValueError("DAT1 header is truncated")
        data_file_id, version, declared_size, block_count, fixup_count = struct.unpack_from("<IIIHH", self.data, 0)
        if data_file_id != DAT1_FILE_ID:
            raise ValueError("invalid DAT1 file id")
        if declared_size > len(self.data):
            raise ValueError("DAT1 declared size is larger than the source file")
        self.data = self.data[:declared_size]
        self.version = version
        self.block_count = block_count
        self.fixup_count = fixup_count
        self.sb_offset = (
            DAT1_HEADER_SIZE
            + block_count * DAT1_BLOCK_TABLE_ENTRY_SIZE
            + fixup_count * DAT1_FIXUP_TABLE_ENTRY_SIZE
        )
        self.entries = []
        self.blocks = {}
        cursor = DAT1_HEADER_SIZE
        for _index in range(block_count):
            name_hash, block_offset, block_size = struct.unpack_from("<III", self.data, cursor)
            self.entries.append((name_hash, block_offset, block_size))
            self.blocks[name_hash] = (block_offset, block_size)
            cursor += DAT1_BLOCK_TABLE_ENTRY_SIZE
        self.fixup_table = self.data[cursor:self.sb_offset]
        first_block = min((off for _hash, off, size in self.entries if size or off), default=len(self.data))
        self.string_buffer = self.data[self.sb_offset:first_block]

    def payload(self, block_hash):
        off, size = self.blocks[block_hash]
        return self.data[off:off + size]


def _material_from_blender(mat, fallback, index, preserve_fallback_identity=False):
    fallback = fallback or {}
    path = str(getattr(mat, "engine_material_path", "") or "")
    if not path:
        mat_name = str(getattr(mat, "name", "") or "")
        if ".material" in mat_name.lower() or "\\" in mat_name or "/" in mat_name:
            path = mat_name
    path = _normalize_material_asset_path(path or fallback.get("path", "") or "")
    fallback_path = _normalize_material_asset_path(fallback.get("path", "") or "")
    if preserve_fallback_identity and fallback_path and _material_paths_match(fallback_path, path):
        mapping = str(fallback.get("mapping", "") or "")
        material_id = int(fallback.get("id", 0) or 0) & U64_MASK
        flags = int(fallback.get("flags", 0) or 0) & U32_MASK
        mapping_hash = int(fallback.get("mapping_hash", 0)) & U32_MASK
        path = fallback_path
    else:
        mapping = str(getattr(mat, "engine_material_mapping_name", "") or fallback.get("mapping", "") or "")
        material_id = _parse_u64(getattr(mat, "get", lambda *_args: None)("engine_material_id"), fallback.get("id", 0))
        flags = int(getattr(mat, "get", lambda *_args: 0)("engine_material_flags", fallback.get("flags", 0)) or 0) & U32_MASK
        mapping_hash = int(fallback.get("mapping_hash", 0)) & U32_MASK
        if mapping:
            mapping_hash = string_crc32(mapping)
    return {
        "index": index,
        "path": path,
        "mapping": mapping,
        "id": material_id or _valid_material_id(index),
        "mapping_hash": mapping_hash,
        "flags": flags,
    }


def _material_asset_path_from_blender(mat):
    path = _normalize_material_asset_path(getattr(mat, "engine_material_path", "") or "")
    if not path:
        mat_name = str(getattr(mat, "name", "") or "")
        if ".material" in mat_name.lower() or "\\" in mat_name or "/" in mat_name:
            path = _normalize_material_asset_path(mat_name)
    return path


def _material_paths_match(left, right):
    left = _normalize_material_asset_path(left or "")
    right = _normalize_material_asset_path(right or "")
    return bool(left and right and left.lower() == right.lower())


def _primary_material_for_object(obj):
    mesh = getattr(obj, "data", None)
    materials = list(getattr(mesh, "materials", []) or []) if mesh else []
    if not materials:
        return None
    used = {}
    for poly in getattr(mesh, "polygons", []) or []:
        used[int(poly.material_index)] = used.get(int(poly.material_index), 0) + 1
    if used:
        index = max(used.items(), key=lambda item: item[1])[0]
        if 0 <= index < len(materials):
            return materials[index]
    return materials[0]


def _set_material_index_prop(mat, material_index):
    try:
        mat["engine_material_index"] = int(material_index)
    except Exception:
        pass


def _set_object_material_index_prop(obj, material_index):
    try:
        obj["engine_material_index"] = int(material_index)
    except Exception:
        pass


def _build_material_entries(mesh_objects, original_materials, export_warnings=None):
    entries = [dict(entry) for entry in original_materials]
    if not entries:
        entries.append({
            "index": 0,
            "path": "",
            "mapping": "default",
            "id": _valid_material_id(0),
            "mapping_hash": string_crc32("default"),
            "flags": 0,
        })
    assigned_indices = []
    claimed_slots = {}
    for obj in mesh_objects:
        mat = _primary_material_for_object(obj)
        fallback_index = int(obj.get("engine_material_index", 0) or 0)
        if not mat:
            assigned_index = _clamp(fallback_index, 0, len(entries) - 1)
            assigned_indices.append(assigned_index)
            _set_object_material_index_prop(obj, assigned_index)
            _append_export_warning(
                export_warnings,
                f"{obj.name} has no Blender material. The original game material was kept. "
                "Assign a material only if you want to replace it.",
            )
            continue

        material_index_prop = mat.get("engine_material_index")
        material_index = None
        try:
            material_index = int(material_index_prop)
        except Exception:
            material_index = None
        
        path = _material_asset_path_from_blender(mat)
        mapping = str(getattr(mat, "engine_material_mapping_name", "") or "")

        if material_index is not None and material_index >= 0:
            existing_path = entries[material_index].get("path", "") if material_index < len(entries) else ""
            slot_is_safe = not existing_path or not path or _material_paths_match(existing_path, path)
            if material_index in claimed_slots and claimed_slots[material_index] != mat.name:
                slot_is_safe = False
            if slot_is_safe:
                while material_index >= len(entries):
                    entries.append({
                        "index": len(entries),
                        "path": "",
                        "mapping": f"material_{len(entries):03d}",
                        "id": _valid_material_id(len(entries)),
                        "mapping_hash": 0,
                        "flags": 0,
                    })
                entries[material_index] = _material_from_blender(
                    mat,
                    entries[material_index],
                    material_index,
                    preserve_fallback_identity=bool(existing_path and path and _material_paths_match(existing_path, path)),
                )
                claimed_slots[material_index] = mat.name
                _set_material_index_prop(mat, material_index)
                _set_object_material_index_prop(obj, material_index)
                assigned_indices.append(material_index)
                continue
            material_index = None

        if not path:
            _append_export_warning(
                export_warnings,
                f"{obj.name}'s material '{mat.name}' is not linked to a game material. The original game "
                "settings were kept. Set Material Asset Path in the Luna Engine Material panel if you want "
                "to replace them.",
            )
        match_index = None
        for idx, entry in enumerate(entries):
            if path and _material_paths_match(entry.get("path", ""), path):
                match_index = idx
                break
            if not path and mapping and entry.get("mapping") == mapping:
                match_index = idx
                break
        if match_index is None:
            match_index = len(entries)
            entries.append(_material_from_blender(mat, None, match_index))
        else:
            entries[match_index] = _material_from_blender(
                mat,
                entries[match_index],
                match_index,
                preserve_fallback_identity=bool(path and _material_paths_match(entries[match_index].get("path", ""), path)),
            )
        _set_material_index_prop(mat, match_index)
        _set_object_material_index_prop(obj, match_index)
        assigned_indices.append(match_index)
    return entries, assigned_indices


def _build_material_block(material_entries, string_pool):
    info_bytes = bytearray()
    runtime_bytes = bytearray()
    for entry in material_entries:
        path_offset = string_pool.add(entry.get("path", ""))
        mapping_offset = string_pool.add(entry.get("mapping", ""))
        info_bytes += struct.pack("<IIII", path_offset, 0, mapping_offset, 0)
    for index, entry in enumerate(material_entries):
        mapping = str(entry.get("mapping", "") or "")
        mapping_hash = int(entry.get("mapping_hash", 0)) & U32_MASK
        if mapping_hash == 0 and mapping:
            mapping_hash = string_crc32(mapping)
        material_id = int(entry.get("id", 0)) & U64_MASK
        if material_id == 0:
            material_id = _valid_material_id(index)
        runtime_bytes += struct.pack("<QII", material_id, mapping_hash, int(entry.get("flags", 0)) & U32_MASK)
    return bytes(info_bytes + runtime_bytes)


def _append_export_warning(warnings, message):
    if warnings is None:
        return
    message = str(message or "").strip()
    if message and message not in warnings:
        warnings.append(message)


def _format_export_warnings(warnings, limit=5):
    shown = list(warnings[:limit])
    if len(warnings) > limit:
        shown.append(
            f"After fixing these, export again to see the remaining {len(warnings) - limit} check(s)"
        )
    return "Export finished. Please check: " + " | ".join(shown)


def _friendly_export_error(exc):
    message = str(exc or "").strip()
    if isinstance(exc, ValueError) and message:
        return message
    return (
        "Something unexpected stopped the export. Re-import the original .model file and try again. "
        "If it keeps happening, share the .blend file and original .model file with the add-on developer."
    )


def _uv_name_slot(name):
    clean = re.sub(r"[^a-z0-9]", "", str(name or "").lower())
    if clean in {"uv0", "uvmap", "map1", "texcoord0", "texturecoordinate0"}:
        return 0
    if clean in {"uv1", "uvmap1", "map2", "texcoord1", "texturecoordinate1"}:
        return 1
    if clean in {"uv2", "uvmap2", "map3", "texcoord2", "texturecoordinate2"}:
        return 2
    return None


def _uv_layer_has_nonzero(layer):
    try:
        for item in layer.data:
            uv = item.uv
            if abs(float(uv.x)) > 1e-7 or abs(1.0 - float(uv.y)) > 1e-7:
                return True
    except Exception:
        pass
    return False


def _zero_uv_layer(layer):
    try:
        for item in layer.data:
            item.uv = (0.0, 1.0)
    except Exception:
        pass


def _uv_layers_match(left_layer, right_layer):
    if not left_layer or not right_layer:
        return False
    try:
        if len(left_layer.data) != len(right_layer.data):
            return False
        for left_item, right_item in zip(left_layer.data, right_layer.data):
            left_uv = left_item.uv
            right_uv = right_item.uv
            if abs(float(left_uv.x) - float(right_uv.x)) > 1e-7:
                return False
            if abs(float(left_uv.y) - float(right_uv.y)) > 1e-7:
                return False
    except Exception:
        return False
    return True


def _ensure_model_uv_layers(obj, warnings=None):
    mesh = getattr(obj, "data", None)
    uv_layers = getattr(mesh, "uv_layers", None) if mesh else None
    result = {
        0: {"layer": None, "source_present": False},
        1: {"layer": None, "source_present": False},
        2: {"layer": None, "source_present": False},
    }
    if uv_layers is None:
        return result

    original_layers = list(uv_layers)
    used = []

    def is_used(layer):
        return any(layer == existing for existing in used)

    def pick_named(slot):
        for layer in original_layers:
            if not is_used(layer) and _uv_name_slot(layer.name) == slot:
                return layer
        return None

    for slot in range(3):
        layer = pick_named(slot)
        if layer is not None:
            used.append(layer)
            result[slot] = {"layer": layer, "source_present": True}

    for slot in range(3):
        if result[slot]["layer"] is not None:
            continue
        if slot < len(original_layers) and not is_used(original_layers[slot]):
            layer = original_layers[slot]
            used.append(layer)
            result[slot] = {"layer": layer, "source_present": True}

    for slot, info in result.items():
        layer = info["layer"]
        if layer is not None and layer.name != f"UV{slot}":
            old_name = layer.name
            layer.name = f"__LunaUV{slot}"
            _append_export_warning(
                warnings,
                f"{obj.name}'s UV map '{old_name}' was renamed to 'UV{slot}' so the game can read it. "
                "No action is needed unless you meant to use a different UV map.",
            )

    for slot, info in result.items():
        layer = info["layer"]
        if layer is None:
            layer = uv_layers.new(name=f"UV{slot}")
            _zero_uv_layer(layer)
            result[slot] = {"layer": layer, "source_present": False}
        layer.name = f"UV{slot}"

    if not original_layers:
        _append_export_warning(
            warnings,
            f"{obj.name} had no UV maps, so blank ones were added. Export can finish, but textures may "
            "look wrong. Unwrap the mesh to UV0, then export again.",
        )
    return result


def _should_export_uv_channel(obj, uv_info, slot, original_flags):
    flag = SUBSET_FLAG_HAS_UV1 if slot == 1 else SUBSET_FLAG_HAS_UV2
    if original_flags & flag:
        return True
    layer = uv_info.get(slot, {}).get("layer")
    prop_name = f"engine_uv{slot}_present"
    if prop_name in obj:
        return bool(obj.get(prop_name, False))
    if not _uv_layer_has_nonzero(layer):
        return False
    uv0_layer = uv_info.get(0, {}).get("layer")
    if uv0_layer is not None and _uv_layers_match(layer, uv0_layer):
        return False
    return bool(uv_info.get(slot, {}).get("source_present", False)) or True


def _uv_layer_by_name_or_index(mesh, name, index):
    try:
        layer = mesh.uv_layers.get(name)
        if layer:
            return layer
    except Exception:
        pass
    try:
        if index < len(mesh.uv_layers):
            return mesh.uv_layers[index]
    except Exception:
        pass
    return None


def _loop_uv(layer, loop_index):
    if not layer:
        return (0.0, 0.0)
    uv = layer.data[loop_index].uv
    return (float(uv.x), 1.0 - float(uv.y))


def _uv_nearly_equal(left, right, threshold=0.0001):
    if left is None or right is None:
        return left is None and right is None
    return (
        abs(float(left[0]) - float(right[0])) <= threshold
        and abs(float(left[1]) - float(right[1])) <= threshold
    )


def _linear_matrix_is_identity(matrix, threshold=1.0e-7):
    for row in range(3):
        for column in range(3):
            expected = 1.0 if row == column else 0.0
            if abs(float(matrix[row][column]) - expected) > threshold:
                return False
    return True


def _source_model_basis(mesh, uv0_layer, basis_coords):
    vertex_count = len(mesh.vertices)
    attributes = {}
    for name in (
        "engine_source_normal_tangent",
        "engine_position_w",
        "engine_source_position",
        "engine_source_uv0_u",
        "engine_source_uv0_v",
    ):
        attribute = mesh.attributes.get(name)
        if attribute is None or attribute.domain != 'POINT' or len(attribute.data) != vertex_count:
            return None
        attributes[name] = attribute.data

    expected_signature = str(mesh.get("engine_source_topology_signature", "") or "")
    current_signature = model_topology_signature(
        int(mesh.loops[loop_index].vertex_index)
        for triangle in mesh.loop_triangles
        for loop_index in triangle.loops
    )
    if not expected_signature or current_signature != expected_signature:
        return None

    expected_normal_signature = str(
        mesh.get("engine_source_corner_normal_signature", "") or ""
    )
    if expected_normal_signature and model_corner_normal_signature(mesh.corner_normals) != expected_normal_signature:
        return None

    source_positions = attributes["engine_source_position"]
    for vertex_index, vertex in enumerate(mesh.vertices):
        current = basis_coords[vertex_index] if basis_coords is not None else vertex.co
        source = source_positions[vertex_index].vector
        if any(abs(float(current[axis]) - float(source[axis])) > 1.0e-7 for axis in range(3)):
            return None

    source_words = attributes["engine_source_normal_tangent"]
    source_uv0_u = attributes["engine_source_uv0_u"]
    source_uv0_v = attributes["engine_source_uv0_v"]
    normal_mismatch_count = 0
    normal_mismatch_limit = max(4, int(len(mesh.loops) * 0.001))
    for loop_index, loop in enumerate(mesh.loops):
        vertex_index = int(loop.vertex_index)
        current_uv = _loop_uv(uv0_layer, loop_index)
        source_uv = (
            float(source_uv0_u[vertex_index].value),
            float(source_uv0_v[vertex_index].value),
        )
        if not _uv_nearly_equal(current_uv, source_uv, threshold=1.0e-7):
            return None

        if expected_normal_signature:
            continue

        source_normal_engine = _decode_packed_normal(int(source_words[vertex_index].value) & U32_MASK)
        source_normal_blender = (
            float(source_normal_engine[0]),
            -float(source_normal_engine[2]),
            float(source_normal_engine[1]),
        )
        current_normal = mesh.corner_normals[loop_index].vector
        current_normal_tuple = (
            float(current_normal.x),
            float(current_normal.y),
            float(current_normal.z),
        )

        if _vec_dot(current_normal_tuple, source_normal_blender) < 0.99999:
            normal_mismatch_count += 1

            if normal_mismatch_count > normal_mismatch_limit:
                return None

    return attributes


def _luna_export_vertex_matches(candidate, normal, tangent, tangent_flip, uv0, uv1, uv2):
    if _vec_dot(candidate["normal"], normal) < (1.0 - 0.002):
        return False
    if _vec_dot(candidate["tangent"], tangent) < (1.0 - 1.6):
        return False
    if (candidate["tangent_flip"] >= 0.0) != (float(tangent_flip) >= 0.0):
        return False
    return (
        _uv_nearly_equal(candidate.get("uv0"), uv0)
        and _uv_nearly_equal(candidate.get("uv1"), uv1)
        and _uv_nearly_equal(candidate.get("uv2"), uv2)
    )


def _vertex_group_joint_map(obj, arm):
    if not arm:
        return {}
    bone_names = {}
    for index, bone in enumerate(getattr(arm.data, "bones", []) or []):
        try:
            engine_index = int(bone.get("engine_joint_index", index))
        except Exception:
            engine_index = index
        bone_names[bone.name] = engine_index
    result = {}
    for group in getattr(obj, "vertex_groups", []) or []:
        if group.name in bone_names:
            result[group.index] = bone_names[group.name]
    return result


def _vertex_weights(mesh_vertex, obj, group_to_joint, source_joint_count):
    weights = []
    if not group_to_joint:
        return weights
    for group_elem in getattr(mesh_vertex, "groups", []) or []:
        joint_index = group_to_joint.get(int(group_elem.group))
        if joint_index is None:
            continue
        weight = float(group_elem.weight)
        if weight > 0.0:
            if source_joint_count is not None and not 0 <= joint_index < source_joint_count:
                raise ValueError(
                    f"{obj.name} has a vertex weighted to a bone that is not in the original skeleton "
                    f"(vertex {mesh_vertex.index}). Remove that weight or use a bone from the imported skeleton, "
                    "then export again."
                )
            weights.append((joint_index, weight))
    weights.sort(key=lambda item: item[1], reverse=True)
    return weights[:12]


def _shape_key_export_data(obj, linear_matrix):
    mesh = obj.data
    shape_keys = getattr(mesh, "shape_keys", None)
    key_blocks = list(getattr(shape_keys, "key_blocks", []) or [])
    if not key_blocks:
        return None, []
    vertex_count = len(mesh.vertices)
    for key in key_blocks:
        if len(key.data) != vertex_count:
            raise ValueError(
                f"Shape key {key.name!r} no longer fits {obj.name}. Delete and recreate that shape key, or "
                "re-import the original model with Import Shape Keys enabled."
            )

    basis_coords = np.empty(vertex_count * 3, dtype=np.float64)
    key_blocks[0].data.foreach_get("co", basis_coords)
    basis_coords = basis_coords.reshape((-1, 3))
    if len(key_blocks) == 1:
        return basis_coords, []

    try:
        metadata = json.loads(str(obj.get("engine_morph_targets_json", "{}") or "{}"))
    except Exception:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}

    transform = np.array(
        [[float(linear_matrix[row][column]) for column in range(3)] for row in range(3)],
        dtype=np.float64,
    )
    targets = []
    for key in key_blocks[1:]:
        if metadata and str(key.name) not in metadata:
            continue
        key_coords = np.empty(vertex_count * 3, dtype=np.float64)
        key.data.foreach_get("co", key_coords)
        blender_deltas = key_coords.reshape((-1, 3)) - basis_coords
        armature_deltas = blender_deltas @ transform.T
        engine_deltas = np.empty_like(armature_deltas)
        engine_deltas[:, 0] = armature_deltas[:, 0]
        engine_deltas[:, 1] = armature_deltas[:, 2]
        engine_deltas[:, 2] = -armature_deltas[:, 1]

        affected = np.flatnonzero(
            np.linalg.norm(engine_deltas, axis=1) >= MORPH_VERTEX_DELTA_EPSILON
        )
        if not len(affected):
            continue

        stored = metadata.get(str(key.name))
        if isinstance(stored, dict) and stored.get("name"):
            target_name = str(stored["name"])
            target_hash = int(stored.get("hash", string_crc32(target_name))) & U32_MASK
            source_target_index = int(stored.get("index", -1))
        else:
            target_name = str(key.name)
            target_hash = string_crc32(target_name)
            source_target_index = -1
        targets.append({
            "name": target_name,
            "hash": target_hash,
            "source_index": source_target_index,
            "source_deltas": {
                int(index): tuple(float(component) for component in engine_deltas[index])
                for index in affected
            },
        })
    return basis_coords, targets


def _finalize_export_morph_topology(vertices, indices, source_targets):
    targets = []
    affected_vertices = set()
    for target in source_targets:
        source_deltas = target["source_deltas"]
        deltas = {}
        for export_index, vertex in enumerate(vertices):
            delta = source_deltas.get(int(vertex["source_index"]))
            if delta is not None:
                deltas[export_index] = delta
        if deltas:
            targets.append({
                "name": target["name"],
                "hash": target["hash"],
                "source_index": target.get("source_index", -1),
                "deltas": deltas,
            })
            affected_vertices.update(deltas)
    if not affected_vertices:
        return vertices, indices, targets, 0

    order = sorted(affected_vertices) + [index for index in range(len(vertices)) if index not in affected_vertices]
    remap = {old_index: new_index for new_index, old_index in enumerate(order)}
    reordered_vertices = [vertices[old_index] for old_index in order]
    reordered_indices = [remap[int(index)] for index in indices]
    for target in targets:
        target["deltas"] = {remap[index]: delta for index, delta in target["deltas"].items()}
    return reordered_vertices, reordered_indices, targets, len(affected_vertices)


def _order_export_vertices_by_control_point(vertices, indices, control_vertex_count):
  
    primary = {}
    duplicates = []
    for old_index, vertex in enumerate(vertices):
        source_index = int(vertex.get("source_index", -1))
        if 0 <= source_index < int(control_vertex_count) and source_index not in primary:
            primary[source_index] = old_index
        else:
            duplicates.append(old_index)
    if len(primary) != int(control_vertex_count):
        # Preserve established behavior for malformed meshes containing unused
        # control points; they cannot safely satisfy direct Luna vertex IDs.
        return vertices, indices
    order = [primary[index] for index in range(int(control_vertex_count))]
    order.extend(sorted(duplicates, key=lambda index: (int(vertices[index].get("source_index", -1)), index)))
    if order == list(range(len(vertices))):
        return vertices, indices
    remap = {old_index: new_index for new_index, old_index in enumerate(order)}
    return [vertices[index] for index in order], [remap[int(index)] for index in indices]


def _export_mesh_vertices(
    obj,
    arm,
    source_joint_count,
    original_flags=0,
    export_warnings=None,
    fallback_morph_targets=None,
):
    mesh = obj.data
    uv_info = _ensure_model_uv_layers(obj, export_warnings)
    mesh.calc_loop_triangles()
    uv0_layer = _uv_layer_by_name_or_index(mesh, "UV0", 0)
    uv1_layer = _uv_layer_by_name_or_index(mesh, "UV1", 1)
    uv2_layer = _uv_layer_by_name_or_index(mesh, "UV2", 2)
    has_uv1 = _should_export_uv_channel(obj, uv_info, 1, original_flags)
    has_uv2 = _should_export_uv_channel(obj, uv_info, 2, original_flags)
    matrix = _safe_matrix_relative_to_armature(arm, obj)
    linear_matrix = matrix.to_3x3()
    try:
        normal_matrix = linear_matrix.inverted().transposed()
    except Exception:
        normal_matrix = linear_matrix
    group_to_joint = _vertex_group_joint_map(obj, arm)
    basis_coords, source_morph_targets = _shape_key_export_data(obj, linear_matrix)
    if not source_morph_targets and fallback_morph_targets:
        source_morph_targets = [
            {
                "name": str(target["name"]),
                "hash": int(target["hash"]) & U32_MASK,
                "source_index": int(target.get("source_index", target.get("index", -1))),
                "deltas": {int(index): tuple(delta) for index, delta in target.get("deltas", {}).items()},
            }
            for target in fallback_morph_targets
        ]

    if not uv0_layer:
        raise ValueError(
            f"{obj.name} needs a UV map called UV0. In Object Data Properties > UV Maps, create or rename "
            "the main texture UV map to UV0, then export again."
        )
    uv0_layer = _uv_layer_by_name_or_index(mesh, "UV0", 0)
    uv1_layer = _uv_layer_by_name_or_index(mesh, "UV1", 1)
    uv2_layer = _uv_layer_by_name_or_index(mesh, "UV2", 2)

    source_position_w = None
    try:
        attr = mesh.attributes.get("engine_position_w")
        if attr and attr.domain == 'POINT' and len(attr.data) == len(mesh.vertices):
            source_position_w = attr.data
    except Exception:
        source_position_w = None

    source_basis = _source_model_basis(mesh, uv0_layer, basis_coords)
    source_basis_packed_exact = bool(source_basis and _linear_matrix_is_identity(linear_matrix))
    try:
        tangent_flip_transform = -1.0 if float(linear_matrix.determinant()) < 0.0 else 1.0
    except Exception:
        tangent_flip_transform = 1.0

    export_vertices = []
    vertex_buckets = {}
    indices = []

    for triangle in mesh.loop_triangles:
        triangle_loops = tuple(int(loop_index) for loop_index in triangle.loops)
        triangle_positions = []
        triangle_uv0s = []
        for loop_index in triangle_loops:
            source_vertex_index = int(mesh.loops[loop_index].vertex_index)
            source_co = (
                mathutils.Vector(basis_coords[source_vertex_index])
                if basis_coords is not None else mesh.vertices[source_vertex_index].co
            )
            triangle_positions.append(_blender_to_engine_vec(matrix @ source_co))
            triangle_uv0s.append(_loop_uv(uv0_layer, loop_index))
        triangle_tangent, triangle_tangent_flip = _luna_triangle_tangent_space(
            triangle_positions,
            triangle_uv0s,
        )
        tri_indices = []
        for triangle_corner, loop_index in enumerate(triangle_loops):
            loop = mesh.loops[loop_index]
            source_vertex_index = int(loop.vertex_index)
            uv0 = _loop_uv(uv0_layer, loop_index)
            uv1 = _loop_uv(uv1_layer, loop_index) if has_uv1 and uv1_layer else None
            uv2 = _loop_uv(uv2_layer, loop_index) if has_uv2 and uv2_layer else None
            # Preserve Blender's split/corner normals. Averaging these by
            # control point rounds deliberately sharp eye and lens frames.
            corner_normal = mesh.corner_normals[loop_index].vector
            normal = _vec_normalize(_blender_to_engine_vec(normal_matrix @ corner_normal))
            packed_source_basis = None
            if source_basis is not None:
                source_word = int(
                    source_basis["engine_source_normal_tangent"][source_vertex_index].value
                ) & U32_MASK
                source_w = int(source_basis["engine_position_w"][source_vertex_index].value)
                source_normal_engine = _decode_packed_normal(source_word)
                source_normal_blender = mathutils.Vector((
                    float(source_normal_engine[0]),
                    -float(source_normal_engine[2]),
                    float(source_normal_engine[1]),
                ))
                normal = _vec_normalize(
                    _blender_to_engine_vec(normal_matrix @ source_normal_blender),
                    fallback=normal,
                )
                source_tangent_engine = _decode_packed_tangent(source_word, source_w)
                source_tangent_blender = mathutils.Vector((
                    float(source_tangent_engine[0]),
                    -float(source_tangent_engine[2]),
                    float(source_tangent_engine[1]),
                ))
                tangent = _vec_normalize(
                    _blender_to_engine_vec(linear_matrix @ source_tangent_blender),
                    fallback=triangle_tangent,
                )
                tangent_flip = (1.0 if source_w >= 0 else -1.0) * tangent_flip_transform
                if source_basis_packed_exact:
                    packed_source_basis = (source_word, source_w)
            else:
                tangent = triangle_tangent
                tangent_flip = triangle_tangent_flip
            extrusion_encoded = 16
            if source_position_w is not None:
                extrusion_encoded = (abs(int(source_position_w[source_vertex_index].value)) >> 10) & 0x1F
            export_index = None
            for candidate_index in vertex_buckets.get(source_vertex_index, ()):
                candidate = export_vertices[candidate_index]
                if _luna_export_vertex_matches(candidate, normal, tangent, tangent_flip, uv0, uv1, uv2):
                    export_index = candidate_index
                    candidate["_normal_sum"] = _vec_add(candidate["_normal_sum"], normal)
                    candidate["_tangent_sum"] = _vec_add(candidate["_tangent_sum"], tangent)
                    candidate["_basis_count"] += 1
                    break
            if export_index is None:
                vertex = mesh.vertices[source_vertex_index]
                export_index = len(export_vertices)
                vertex_buckets.setdefault(source_vertex_index, []).append(export_index)
                export_vertices.append({
                    "source_index": source_vertex_index,
                    "co": triangle_positions[triangle_corner],
                    "normal": normal,
                    "tangent": tangent,
                    "tangent_flip": tangent_flip,
                    "extrusion_encoded": extrusion_encoded,
                    "_normal_sum": normal,
                    "_tangent_sum": tangent,
                    "_basis_count": 1,
                    "_packed_source_basis": packed_source_basis,
                    "uv0": uv0,
                    "uv1": uv1,
                    "uv2": uv2,
                    "weights": _vertex_weights(vertex, obj, group_to_joint, source_joint_count),
                })
            tri_indices.append(export_index)
        indices.extend(tri_indices)


    for vertex in export_vertices:
        normal = _vec_normalize(vertex.pop("_normal_sum"))
        tangent = _vec_normalize(vertex.pop("_tangent_sum"), fallback=(1.0, 0.0, 0.0))
        vertex.pop("_basis_count", None)
        vertex["normal"] = normal
        vertex["tangent"] = tangent
        packed_source_basis = vertex.pop("_packed_source_basis", None)
        tangent_flip = vertex.pop("tangent_flip")
        extrusion_encoded = vertex.pop("extrusion_encoded")
        if packed_source_basis is not None:
            vertex["normal_tangent"] = int(packed_source_basis[0]) & U32_MASK
            vertex["position_w"] = int(packed_source_basis[1])
        else:
            normal_tangent, tangent_y = _pack_normal_tangent(normal, tangent)
            vertex["normal_tangent"] = normal_tangent
            vertex["position_w"] = _pack_position_w(
                tangent_y,
                tangent_flip,
                extrusion_encoded,
            )

    export_vertices, indices = _order_export_vertices_by_control_point(
        export_vertices,
        indices,
        len(mesh.vertices),
    )

    export_vertices, indices, morph_targets, anim_vert_count = _finalize_export_morph_topology(
        export_vertices,
        indices,
        source_morph_targets,
    )
    return export_vertices, indices, bool(has_uv1), bool(has_uv2), morph_targets, anim_vert_count


def _split_export_vertex_chunks(vertices, indices, max_vertices=MODEL_SPLIT_VERTEX_TARGET):
    chunks = []
    chunk_vertices = []
    chunk_indices = []
    remap = {}
    max_vertices = max(3, min(int(max_vertices), MODEL_MAX_VERTEX_COUNT))

    triangles = []
    for tri_order, tri_start in enumerate(range(0, len(indices), 3)):
        tri = indices[tri_start:tri_start + 3]
        if len(tri) < 3:
            continue
        coords = [vertices[int(index)]["co"] for index in tri]
        centroid = (
            (coords[0][0] + coords[1][0] + coords[2][0]) / 3.0,
            (coords[0][1] + coords[1][1] + coords[2][1]) / 3.0,
            (coords[0][2] + coords[1][2] + coords[2][2]) / 3.0,
        )
        triangles.append((-centroid[1], centroid[0], centroid[2], tri_order, tri))
    triangles.sort()

    def flush_chunk():
        nonlocal chunk_vertices, chunk_indices, remap
        if chunk_vertices and chunk_indices:
            chunks.append((chunk_vertices, chunk_indices))
        chunk_vertices = []
        chunk_indices = []
        remap = {}

    for _height_key, _x_key, _z_key, _tri_order, tri in triangles:
        missing = []
        for source_index in tri:
            if source_index not in remap and source_index not in missing:
                missing.append(source_index)
        if chunk_vertices and len(chunk_vertices) + len(missing) > max_vertices:
            flush_chunk()

        for source_index in tri:
            mapped = remap.get(source_index)
            if mapped is None:
                mapped = len(chunk_vertices)
                remap[source_index] = mapped
                chunk_vertices.append(vertices[source_index])
            chunk_indices.append(mapped)

        if len(chunk_vertices) > max_vertices:
            raise ValueError(
                "One face is too large for the game format. Apply the mesh modifiers, triangulate the mesh, "
                "and split very large geometry into smaller objects before exporting again."
            )

    flush_chunk()
    return chunks


def _uv_log_for_values(vertices, channel_name):
    max_abs = 1.0
    for vertex in vertices:
        uv = vertex.get(channel_name)
        if uv is None:
            continue
        max_abs = max(max_abs, abs(float(uv[0])), abs(float(uv[1])))
    unbounded = int(round(max_abs * MODEL_UV_FLOAT_TO_FIXED_BASE)) >> 15
    return _clamp(unbounded.bit_length() if unbounded > 0 else 0, 0, 15)


def _uv_scale(log_value):
    return float(1 << int(log_value)) / MODEL_UV_FLOAT_TO_FIXED_BASE


def _pack_uv(uv, log_value):
    scale = _uv_scale(log_value)
    return (
        _clamp_i16(float(uv[0]) / scale),
        _clamp_i16(float(uv[1]) / scale),
    )


def _encode_azimuthal(vector):
    vector = _vec_normalize(vector)
    inv_f = 1.0 / math.sqrt(abs(vector[2]) * 4.0 + 4.0)
    return (
        vector[0] * inv_f + 0.5,
        vector[1] * inv_f + 0.5,
        0.0 if vector[2] < 0.0 else 1.0,
    )


def _pack_normal_tangent(normal, tangent):
    normal_azim = _encode_azimuthal(normal)
    tangent_azim = _encode_azimuthal(tangent)
    norm_tan_z = normal_azim[2] * (2.0 / 3.0) + tangent_azim[2] * (1.0 / 3.0)
    nx = _clamp(_round_engine(normal_azim[0] * 1023.0), 0, 1023)
    ny = _clamp(_round_engine(normal_azim[1] * 1023.0), 0, 1023)
    tx = _clamp(_round_engine(tangent_azim[0] * 1023.0), 0, 1023)
    ty = _clamp(_round_engine(tangent_azim[1] * 1023.0), 0, 1023)
    nz_tz = _clamp(_round_engine(norm_tan_z * 3.0), 0, 3)
    packed = int(nx) | (int(ny) << 10) | (int(tx) << 20) | (int(nz_tz) << 30)
    return packed, int(ty)


def _packed_normal_key(normal):
    normal_azim = _encode_azimuthal(normal)
    return (
        _clamp(_round_engine(normal_azim[0] * 1023.0), 0, 1023),
        _clamp(_round_engine(normal_azim[1] * 1023.0), 0, 1023),
        normal_azim[2] >= 0.5,
    )


def _pack_position_w(tangent_y, tangent_flip, extrusion_encoded=16):
    magnitude = (int(tangent_y) & 0x3FF) | ((int(extrusion_encoded) & 0x1F) << 10)
    if float(tangent_flip) >= 0.0:
        return magnitude
    return -1 if magnitude == 0 else -magnitude


def _normalize_skin_weights(weights):
    if not weights:
        return [(0, 256)]
    weights = weights[:12]
    total = sum(max(0.0, weight) for _joint, weight in weights)
    if total <= 1e-8:
        return [(int(weights[0][0]), 256)]
    scaled = []
    running = 0
    for joint, weight in weights:
        exact = max(0.0, weight) * 256.0 / total
        base = int(math.floor(exact))
        scaled.append([int(joint), base, exact - base])
        running += base
    remainder = 256 - running
    scaled.sort(key=lambda item: item[2], reverse=True)
    for index in range(abs(remainder)):
        scaled[index % len(scaled)][1] += 1 if remainder > 0 else -1
    scaled.sort(key=lambda item: item[1], reverse=True)
    result = [(joint, _clamp(weight, 0, 256)) for joint, weight, _frac in scaled if weight > 0]
    return result or [(int(weights[0][0]), 256)]


def _skin_joint_entries(weights):
    entries = [(int(joint), int(weight)) for joint, weight in weights if int(weight) > 0]
    return entries or [(int(weights[0][0]) if weights else 0, 256)]


def _split_full_influence_joints(joints, is_16bit, cluster_joint_count):
    if cluster_joint_count > 1 and len(joints) >= 1 and joints[0][1] == 256:
        joint_index = joints[0][0]
        joints[0] = (joint_index, 128)
        joints.insert(1, (joint_index if is_16bit else 0, 128))
        return True
    return False


def _insert_bridge_indices(joints, joint_count_max):
    joint_count = len(joints)
    index = 0
    while index < joint_count:
        joint_index, weight = joints[index]
        if joint_index <= SKIN_UINT8_MAX:
            index += 1
            continue
        joints.insert(index + 1, (joint_index - SKIN_UINT8_MAX, weight))
        joints[index] = (SKIN_UINT8_MAX, 0)
        joint_count += 1
        if joint_count >= joint_count_max:
            break
        index += 1


def _prepare_cluster_skin(normalized_weights):
    cluster_vertex_count = len(normalized_weights)
    source_joints = [_skin_joint_entries(weights) for weights in normalized_weights]

    joint_min = None
    joint_count_max = 1
    for joints in source_joints:
        for joint_index, weight in joints:
            if weight > 0:
                joint_min = joint_index if joint_min is None else min(joint_min, joint_index)
        joint_count_max = max(joint_count_max, len(joints))
    if joint_min is None:
        joint_min = 0
    joint_offset = min((joint_min // SKIN_JOINT_OFFSET_STEP) * SKIN_JOINT_OFFSET_STEP, SKIN_JOINT_OFFSET_MAX)
    if joint_count_max == 0 or joint_count_max > 12:
        raise ValueError(
            "Some vertices use more than 12 bone weights. In Weight Paint mode, limit each vertex to 12 "
            "bones or fewer, normalize the weights, then export again."
        )

    prepared = []
    is_16bit = False
    for joints in source_joints:
        working = [(joint_index - joint_offset, weight) for joint_index, weight in joints]
        working.sort(key=lambda item: item[0])
        if working[0][0] > SKIN_UINT8_MAX:
            is_16bit = True
            break

        incremental = list(working)
        for index in range(len(incremental) - 1, 0, -1):
            incremental[index] = (incremental[index][0] - incremental[index - 1][0], incremental[index][1])

        if len(incremental) < joint_count_max:
            _insert_bridge_indices(incremental, joint_count_max)

        if any(joint_index > SKIN_UINT8_MAX for joint_index, _weight in incremental):
            is_16bit = True
            break
        prepared.append(incremental)

    if is_16bit:
        joint_offset = 0
        prepared = []
        for joints in source_joints:
            absolute = sorted(joints, key=lambda item: item[0])
            prepared.append(list(absolute))
        if joint_count_max > 2:
            joint_count_max = min(12, _align(joint_count_max, 4))

    for index, joints in enumerate(prepared):
        _split_full_influence_joints(joints, is_16bit, joint_count_max)
        while len(joints) < joint_count_max:
            joints.append((0, 0))
        prepared[index] = joints[:joint_count_max]

    return is_16bit, joint_offset, joint_count_max, prepared[:cluster_vertex_count]


def _serialize_cluster_skin_data(prepared, is_16bit, joint_count_max):
    cluster_bytes = bytearray()
    for joints in prepared:
        for joint_index, weight in joints[:joint_count_max]:
            if is_16bit:
                cluster_bytes += struct.pack("<H", int(joint_index) & 0xFFFF)
                if joint_count_max > 1:
                    cluster_bytes += struct.pack("<B", int(weight) & 0xFF)
            else:
                cluster_bytes += struct.pack("<B", int(joint_index) & 0xFF)
                if joint_count_max > 1:
                    cluster_bytes += struct.pack("<B", int(weight) & 0xFF)
    pad = (-len(cluster_bytes)) & 3
    if pad:
        cluster_bytes += b"\x00" * pad
    return bytes(cluster_bytes)


def _build_skin_sections(vertices, force_skin, anim_cluster_count=0):
    if not force_skin:
        return b"", b""
    skin_data = bytearray()
    cluster_headers = bytearray()
    for cluster_index, cluster_start in enumerate(range(0, len(vertices), SKIN_CLUSTER_VERTEX_COUNT)):
        cluster_vertices = vertices[cluster_start:cluster_start + SKIN_CLUSTER_VERTEX_COUNT]
        normalized_weights = [_normalize_skin_weights(vertex.get("weights", [])) for vertex in cluster_vertices]
        influence_count = max(1, min(12, max(len(weights) for weights in normalized_weights)))

        if influence_count > 1:
            normalized_weights = [
                [(weights[0][0], 128), (weights[0][0], 128)]
                if len(weights) == 1 and weights[0][1] == 256 else weights
                for weights in normalized_weights
            ]
            influence_count = max(2, min(12, max(len(weights) for weights in normalized_weights)))

        is_16bit, joint_offset, joint_count_max, prepared = _prepare_cluster_skin(normalized_weights)

        _align_buffer(skin_data, SKIN_CLUSTER_WORD_BYTES)
        data_offset4 = len(skin_data) // SKIN_CLUSTER_WORD_BYTES
        if data_offset4 > SKIN_CLUSTER_OFFSET_MASK:
            raise ValueError(
                "This weighted mesh is too large for one game mesh part. Split it into smaller objects, keep "
                "their Armature parent and weights, then export again."
            )
        header = data_offset4 | ((joint_count_max - 1) << SKIN_CLUSTER_INFLUENCE_SHIFT)
        if is_16bit:
            header |= SKIN_CLUSTER_FULL_INDEX_BIT
        else:
            header |= (joint_offset // SKIN_JOINT_OFFSET_STEP) << SKIN_CLUSTER_JOINT_OFFSET_SHIFT
        if cluster_index < int(anim_cluster_count):
            header |= SKIN_CLUSTER_ANIM_VERT_BIT
        cluster_headers += struct.pack("<I", header)
        skin_data += _serialize_cluster_skin_data(prepared, is_16bit, joint_count_max)
    return bytes(skin_data), bytes(cluster_headers)


def _triangle_area(a, b, c):
    return 0.5 * _vec_len(_vec_cross(_vec_sub(b, a), _vec_sub(c, a)))


def _triangle_uv_area(a, b, c):
    return abs(
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    ) * 0.5


def _repair_missing_skin_weights(vertices, indices):
    missing = {index for index, vertex in enumerate(vertices) if not vertex.get("weights")}
    if not missing:
        return 0, 0

    source_vertices = {
        int(vertices[index].get("source_index", index))
        for index in missing
    }
    adjacency = [set() for _vertex in vertices]
    for index in range(0, len(indices) - 2, 3):
        a, b, c = (int(indices[index]), int(indices[index + 1]), int(indices[index + 2]))
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))

    while missing:
        updates = {}
        for vertex_index in missing:
            neighbor_weights = [
                vertices[neighbor].get("weights", [])
                for neighbor in adjacency[vertex_index]
                if vertices[neighbor].get("weights")
            ]
            if not neighbor_weights:
                continue
            totals = {}
            for weights in neighbor_weights:
                for joint, weight in weights:
                    totals[int(joint)] = totals.get(int(joint), 0.0) + float(weight)
            divisor = float(len(neighbor_weights))
            updates[vertex_index] = sorted(
                ((joint, weight / divisor) for joint, weight in totals.items() if weight > 0.0),
                key=lambda item: item[1],
                reverse=True,
            )[:12]
        if not updates:
            break
        for vertex_index, weights in updates.items():
            vertices[vertex_index]["weights"] = weights
        missing.difference_update(updates)

    fallback_count = len(missing)
    if missing:
        joint_totals = {}
        for vertex in vertices:
            for joint, weight in vertex.get("weights", []):
                joint_totals[int(joint)] = joint_totals.get(int(joint), 0.0) + float(weight)
        fallback_joint = max(joint_totals, key=joint_totals.get) if joint_totals else 0
        for vertex_index in missing:
            vertices[vertex_index]["weights"] = [(fallback_joint, 1.0)]

    return len(source_vertices), fallback_count


def _fit_subset_mpu(vertices, requested_mpu):
    coords = [vertex["co"] for vertex in vertices]
    mins = [min(coord[axis] for coord in coords) for axis in range(3)]
    maxs = [max(coord[axis] for coord in coords) for axis in range(3)]
    center = [(mins[axis] + maxs[axis]) * 0.5 for axis in range(3)]
    extents = [(maxs[axis] - mins[axis]) * 0.5 for axis in range(3)]
    radius = max((_vec_len(_vec_sub(coord, center)) for coord in coords), default=0.0)
    residual_limit = 32767 - (1 << (SUBSET_CENTER_LOG_SCALE - 1))
    required_mpu = max(float(requested_mpu), 1e-12)
    for axis in range(3):
        required_mpu = max(
            required_mpu,
            extents[axis] / float(residual_limit),
            abs(center[axis]) / float(32767 * (1 << SUBSET_CENTER_LOG_SCALE)),
            extents[axis] / float(255 * (1 << SUBSET_CENTER_LOG_SCALE)),
        )
    required_mpu = max(
        required_mpu,
        radius / float(255 * (2 << SUBSET_CENTER_LOG_SCALE)),
    )
    return math.nextafter(required_mpu, math.inf)


def _subset_bounds(vertices, mpu):
    coords = [vertex["co"] for vertex in vertices]
    mins = [min(coord[axis] for coord in coords) for axis in range(3)]
    maxs = [max(coord[axis] for coord in coords) for axis in range(3)]
    center = tuple((mins[axis] + maxs[axis]) * 0.5 for axis in range(3))
    extents = tuple((maxs[axis] - mins[axis]) * 0.5 for axis in range(3))
    radius = max((_vec_len(_vec_sub(coord, center)) for coord in coords), default=0.0)

    units = [
        [int(round(coord[axis] / mpu)) for coord in coords]
        for axis in range(3)
    ]
    needs_origin = any(
        min(units[axis]) < -32768 or max(units[axis]) > 32767
        for axis in range(3)
    )
    origin_packed = [
        _clamp_i16(center[axis] / (mpu * float(1 << SUBSET_CENTER_LOG_SCALE)))
        for axis in range(3)
    ]
    origin_units = [
        packed << SUBSET_CENTER_LOG_SCALE if needs_origin else 0
        for packed in origin_packed
    ]

    extents_packed = (
        int(_clamp(math.ceil(extents[0] / (mpu * float(1 << SUBSET_CENTER_LOG_SCALE))), 0, 255)),
        int(_clamp(math.ceil(extents[1] / (mpu * float(1 << SUBSET_CENTER_LOG_SCALE))), 0, 255)),
        int(_clamp(math.ceil(extents[2] / (mpu * float(1 << SUBSET_CENTER_LOG_SCALE))), 0, 255)),
        int(_clamp(math.ceil(radius / (mpu * float(2 << SUBSET_CENTER_LOG_SCALE))), 0, 255)),
    )
    extents_word = (
        extents_packed[0]
        | (extents_packed[1] << 8)
        | (extents_packed[2] << 16)
        | (extents_packed[3] << 24)
    )
    return center, extents, radius, origin_units, origin_packed, extents_word, needs_origin


def _build_subset_geometry_from_data(
    obj,
    arm,
    material_index,
    original_record,
    custom_stream_index,
    vertices,
    indices,
    has_uv1,
    has_uv2,
    anim_vert_count=0,
    export_warnings=None,
):
    original_flags = 0
    if original_record:
        original_flags = struct.unpack_from("<H", original_record, MODEL_SUBSET_FLAGS_OFFSET)[0]
    if len(vertices) > MODEL_MAX_VERTEX_COUNT:
        raise ValueError(
            f"{obj.name} is too large for one game mesh part. Split it into smaller meshes with fewer than "
            f"{MODEL_MAX_VERTEX_COUNT} export vertices each, then try again."
        )
    if any(index >= MODEL_MAX_VERTEX_COUNT for index in indices):
        raise ValueError(
            f"{obj.name} is too large for one game mesh part. Split it into smaller meshes, then try again."
        )

    mpu = float(obj.get("engine_mpu", arm.get("engine_mpu", MODEL_DEFAULT_MPU) if arm else MODEL_DEFAULT_MPU) or MODEL_DEFAULT_MPU)
    if mpu <= 0.0:
        mpu = MODEL_DEFAULT_MPU
    mpu = _fit_subset_mpu(vertices, mpu)

    uv0_log = _uv_log_for_values(vertices, "uv0")
    uv1_log = _uv_log_for_values(vertices, "uv1") if has_uv1 else 0
    uv2_log = _uv_log_for_values(vertices, "uv2") if has_uv2 else 0
    uv_log_scales = int(uv0_log) | (int(uv1_log) << 4) | (int(uv2_log) << 8)

    center, extents, radius, origin_units, origin_packed, extents_word, needs_origin = _subset_bounds(vertices, mpu)

    std_vertices = bytearray()
    for vertex in vertices:
        co = vertex["co"]
        packed_pos = (
            int(_clamp(int(round(co[0] / mpu)) - origin_units[0], -32768, 32767)),
            int(_clamp(int(round(co[1] / mpu)) - origin_units[1], -32768, 32767)),
            int(_clamp(int(round(co[2] / mpu)) - origin_units[2], -32768, 32767)),
        )
        normal_tangent = int(vertex["normal_tangent"])
        uv0 = _pack_uv(vertex.get("uv0", (0.0, 0.0)), uv0_log)
        std_vertices += struct.pack(
            "<hhhhIhh",
            packed_pos[0],
            packed_pos[1],
            packed_pos[2],
            int(vertex["position_w"]),
            normal_tangent,
            uv0[0],
            uv0[1],
        )

    geom = bytearray()
    vertex_std_offset = 0
    geom += std_vertices

    vertex_uv12_offset = 0
    if has_uv1 or has_uv2:
        _align_buffer(geom, 4)
        vertex_uv12_offset = len(geom)
        for vertex in vertices:
            if has_uv1:
                geom += struct.pack("<hh", *_pack_uv(vertex.get("uv1") or (0.0, 0.0), uv1_log))
            if has_uv2:
                geom += struct.pack("<hh", *_pack_uv(vertex.get("uv2") or (0.0, 0.0), uv2_log))

    _align_buffer(geom, 2)
    index_data_offset = len(geom)
    geom += struct.pack(f"<{len(indices)}H", *[int(index) & 0xFFFF for index in indices])

    force_skin = bool(original_flags & SUBSET_FLAG_SKINNED) or any(vertex.get("weights") for vertex in vertices)
    if force_skin:
        repaired_count, fallback_count = _repair_missing_skin_weights(vertices, indices)
        if repaired_count:
            detail = (
                f"; {fallback_count} disconnected export vertices used the object's dominant bone"
                if fallback_count else ""
            )
            _append_export_warning(
                export_warnings,
                f"{obj.name}: automatically repaired missing bone weights on {repaired_count} vertices{detail}.",
            )
    anim_vert_count = int(anim_vert_count)
    if not 0 <= anim_vert_count <= len(vertices):
        raise ValueError(
            f"{obj.name}'s saved facial-animation setup no longer matches the mesh. Re-import the original "
            "model with Import Shape Keys enabled, then repeat your edits."
        )
    anim_cluster_count = _align(anim_vert_count, SKIN_CLUSTER_VERTEX_COUNT) // SKIN_CLUSTER_VERTEX_COUNT
    skin_data, cluster_headers = _build_skin_sections(vertices, force_skin, anim_cluster_count)
    vertex_skin_offset = 0
    skin_cluster_offset = 0
    if force_skin:
        _align_buffer(geom, SKIN_CLUSTER_WORD_BYTES)
        vertex_skin_offset = len(geom)
        geom += skin_data
        _align_buffer(geom, SKIN_CLUSTER_WORD_BYTES)
        skin_cluster_offset = len(geom)
        geom += cluster_headers

    surface_area = 0.0
    uv_area = 0.0
    longest_edge = 0.0
    for i in range(0, len(indices), 3):
        a = vertices[indices[i]]
        b = vertices[indices[i + 1]]
        c = vertices[indices[i + 2]]
        surface_area += _triangle_area(a["co"], b["co"], c["co"])
        uv_area += _triangle_uv_area(a.get("uv0", (0.0, 0.0)), b.get("uv0", (0.0, 0.0)), c.get("uv0", (0.0, 0.0)))
        longest_edge = max(
            longest_edge,
            _vec_len(_vec_sub(a["co"], b["co"])),
            _vec_len(_vec_sub(b["co"], c["co"])),
            _vec_len(_vec_sub(c["co"], a["co"])),
        )

    record = bytearray(original_record if original_record else b"\x00" * MODEL_SUBSET_RECORD_SIZE)
    if len(record) < MODEL_SUBSET_RECORD_SIZE:
        record += b"\x00" * (MODEL_SUBSET_RECORD_SIZE - len(record))
    flags = (original_flags & ~SUBSET_EXPORT_FLAG_CLEAR_MASK)
    if force_skin:
        flags |= SUBSET_FLAG_SKINNED
    if has_uv1:
        flags |= SUBSET_FLAG_HAS_UV1
    if has_uv2:
        flags |= SUBSET_FLAG_HAS_UV2
    if anim_vert_count:
        flags |= SUBSET_FLAG_HAS_ANIM_VERT
    if needs_origin:
        flags |= SUBSET_FLAG_HAS_ORIGIN_OFFSET

    struct.pack_into("<I", record, MODEL_SUBSET_INDEX_COUNT_OFFSET, len(indices))
    struct.pack_into("<I", record, MODEL_SUBSET_VERTEX_COUNT_OFFSET, len(vertices))
    struct.pack_into("<I", record, 8, 0)
    struct.pack_into("<I", record, MODEL_SUBSET_INDEX_DATA_OFFSET, index_data_offset)
    struct.pack_into("<I", record, 16, 0x0000FFFF)
    struct.pack_into("<H", record, MODEL_SUBSET_FLAGS_OFFSET, flags)
    struct.pack_into("<H", record, MODEL_SUBSET_UV_LOG_OFFSET, uv_log_scales)
    struct.pack_into("<f", record, MODEL_SUBSET_MPU_OFFSET, mpu)
    struct.pack_into("<H", record, MODEL_SUBSET_MATERIAL_INDEX_OFFSET, int(material_index) & 0xFFFF)
    if original_record:
        try:
            original_index_count, original_vertex_count = struct.unpack_from("<II", original_record, 0)
        except Exception:
            original_index_count = original_vertex_count = -1
        if original_index_count == len(indices) and original_vertex_count == len(vertices):
            surface_area, uv_area = struct.unpack_from("<ff", original_record, MODEL_SUBSET_SURFACE_AREA_OFFSET)
            longest_edge = float(struct.unpack_from("<H", original_record, MODEL_SUBSET_LONGEST_EDGE_OFFSET)[0]) * mpu

    struct.pack_into("<f", record, MODEL_SUBSET_SURFACE_AREA_OFFSET, float(surface_area))
    struct.pack_into("<f", record, MODEL_SUBSET_UV_AREA_OFFSET, float(uv_area))
    struct.pack_into("<f", record, MODEL_SUBSET_FADE_OUT_DIST_OFFSET, 0.0)
    struct.pack_into("<iii", record, MODEL_SUBSET_OBJ_CENTER_OFFSET, int(origin_packed[0]), int(origin_packed[1]), int(origin_packed[2]))
    struct.pack_into("<I", record, MODEL_SUBSET_OBJ_EXTENTS_OFFSET, extents_word)
    struct.pack_into("<I", record, MODEL_SUBSET_VERTEX_STD_OFFSET, vertex_std_offset)
    struct.pack_into("<I", record, MODEL_SUBSET_VERTEX_UV12_OFFSET, vertex_uv12_offset)
    struct.pack_into("<I", record, 72, 0)
    struct.pack_into("<I", record, 76, vertex_skin_offset)
    struct.pack_into("<I", record, 80, skin_cluster_offset)
    struct.pack_into("<I", record, 84, custom_stream_index)
    struct.pack_into("<I", record, MODEL_SUBSET_BASE_OFFSET, 0)
    struct.pack_into("<I", record, 92, len(geom))
    struct.pack_into("<I", record, 96, anim_vert_count)
    struct.pack_into("<H", record, 100, anim_cluster_count)
    struct.pack_into("<H", record, 102, 1)
    struct.pack_into("<H", record, MODEL_SUBSET_LONGEST_EDGE_OFFSET, _clamp_u16(longest_edge / max(mpu, 1e-9)))

    stats = {
        "vertex_count": len(vertices),
        "index_count": len(indices),
        "custom_stream_count": len(vertices),
        "skinned": force_skin,
        "center": center,
        "extents": extents,
        "radius": radius,
        "mpu": mpu,
        "anim_vert_count": anim_vert_count,
        "anim_cluster_count": anim_cluster_count,
        "vertices": vertices,
    }
    return bytes(record), bytes(geom), stats


def _build_subset_geometry_chunks(
    obj,
    arm,
    material_index,
    original_record,
    source_joint_count,
    export_warnings=None,
    fallback_morph_targets=None,
):
    original_flags = 0
    if original_record:
        original_flags = struct.unpack_from("<H", original_record, MODEL_SUBSET_FLAGS_OFFSET)[0]

    vertices, indices, has_uv1, has_uv2, morph_targets, anim_vert_count = _export_mesh_vertices(
        obj,
        arm,
        source_joint_count,
        original_flags=original_flags,
        export_warnings=export_warnings,
        fallback_morph_targets=fallback_morph_targets,
    )
    if not vertices or not indices:
        raise ValueError(
            f"{obj.name} has no faces that can be exported. Add or triangulate faces, then export again."
        )
    if len(vertices) > MODEL_MAX_VERTEX_COUNT or any(index >= MODEL_MAX_VERTEX_COUNT for index in indices):
        raise ValueError(
            f"{obj.name} is too large for one game mesh part ({len(vertices)} export vertices). Split it into "
            f"smaller meshes with fewer than {MODEL_MAX_VERTEX_COUNT} export vertices each, then try again."
        )
    return [(vertices, indices, has_uv1, has_uv2, morph_targets, anim_vert_count)]


def _build_geometry_and_subset_blocks(
    mesh_objects,
    arm,
    material_indices,
    template,
    source_joint_count,
    export_warnings=None,
    source_morph_targets_by_subset=None,
):
    subset_block_hash = BLOCK_HASHES["ModelSubset"]
    original_subset_block = template.payload(subset_block_hash) if subset_block_hash in template.blocks else b""
    original_count = len(original_subset_block) // MODEL_SUBSET_RECORD_SIZE

    subset_records = []
    geom_buffer = bytearray()
    stats = []
    subset_index_map = {}
    morph_targets_by_name = {}
    custom_stream_index = 0
    for index, obj in enumerate(mesh_objects):
        original_record = b""
        try:
            old_index_value = obj.get("engine_subset_index", index)
            old_index = int(index if old_index_value is None else old_index_value)
        except Exception:
            old_index = index
        if 0 <= old_index < original_count:
            start = old_index * MODEL_SUBSET_RECORD_SIZE
            original_record = original_subset_block[start:start + MODEL_SUBSET_RECORD_SIZE]

        chunks = _build_subset_geometry_chunks(
            obj,
            arm,
            material_indices[index],
            original_record,
            source_joint_count,
            export_warnings=export_warnings,
            fallback_morph_targets=(source_morph_targets_by_subset or {}).get(int(old_index), []),
        )
        mapped_indices = []
        for chunk_vertices, chunk_indices, has_uv1, has_uv2, chunk_morph_targets, anim_vert_count in chunks:
            _align_buffer(geom_buffer, DAT1_BLOCK_ALIGN)
            geom_base = len(geom_buffer)
            record, geom, subset_stats = _build_subset_geometry_from_data(
                obj,
                arm,
                material_indices[index],
                original_record,
                custom_stream_index,
                chunk_vertices,
                chunk_indices,
                has_uv1,
                has_uv2,
                anim_vert_count=anim_vert_count,
                export_warnings=export_warnings,
            )
            record = bytearray(record)
            struct.pack_into("<I", record, MODEL_SUBSET_BASE_OFFSET, geom_base)
            geom_buffer += geom
            generated_subset_index = len(subset_records)
            mapped_indices.append(generated_subset_index)
            subset_records.append(bytes(record))
            subset_stats["indices"] = chunk_indices
            subset_stats["source_subset_index"] = int(old_index)
            stats.append(subset_stats)
            for target in chunk_morph_targets:
                name = str(target["name"])
                existing = morph_targets_by_name.get(name)
                if existing is None:
                    existing = {
                        "name": name,
                        "hash": int(target["hash"]) & U32_MASK,
                        "source_index": int(target.get("source_index", -1)),
                        "subsets": [],
                    }
                    morph_targets_by_name[name] = existing
                elif int(existing["hash"]) != (int(target["hash"]) & U32_MASK):
                    raise ValueError(
                        f"Facial shape {name!r} is registered differently on separate meshes. Remove and "
                        "register that shape again with the same name on every mesh, then export again."
                    )
                elif int(existing.get("source_index", -1)) != int(target.get("source_index", -1)):
                    raise ValueError(
                        f"Facial shape {name!r} comes from different source slots on separate meshes. Re-import "
                        "with Import Shape Keys enabled and register matching shapes, then export again."
                    )
                existing["subsets"].append({
                    "subset_index": generated_subset_index,
                    "deltas": target["deltas"],
                })
            custom_stream_index += _align(subset_stats["custom_stream_count"], 2)
        subset_index_map[int(old_index)] = mapped_indices
    return (
        b"".join(subset_records),
        bytes(geom_buffer),
        stats,
        subset_index_map,
        list(morph_targets_by_name.values()),
    )


def _json_list_from_idprop(owner, key):
    try:
        data = json.loads(str(owner.get(key, "[]") or "[]"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _mapped_subset_ids(source_ids, subset_index_map, subset_count, allow_direct=False):
    result = []
    for value in source_ids:
        try:
            source_id = int(value)
        except Exception:
            continue
        mapped = subset_index_map.get(source_id)
        if mapped is None and allow_direct and 0 <= source_id < subset_count:
            mapped = source_id
        if mapped is None:
            continue
        mapped_values = mapped if isinstance(mapped, (list, tuple, set)) else [mapped]
        for mapped_value in mapped_values:
            try:
                mapped_index = int(mapped_value)
            except Exception:
                continue
            if 0 <= mapped_index < subset_count and mapped_index not in result:
                result.append(mapped_index)
    return result


def _build_look_group_block(groups, look_count, string_pool):
    if not groups:
        groups = [{
            "name": "default",
            "name_hash": string_crc32("default"),
            "look_indices": list(range(look_count)),
        }]

    records = bytearray()
    indices = bytearray()
    records_base = 1
    indices_base = len(groups) * MODEL_LOOK_GROUP_SIZE
    for group_index, group in enumerate(groups):
        raw_indices = group.get("look_indices", [])
        look_indices = []
        for value in raw_indices:
            try:
                look_index = int(value)
            except Exception:
                continue
            if 0 <= look_index < look_count and look_index not in look_indices:
                look_indices.append(look_index)

        name = str(group.get("name", "") or f"Look Group {group_index}")
        name_hash = int(group.get("name_hash", 0) or 0) & U32_MASK
        if not name_hash:
            name_hash = string_crc32(name)
        name_offset = string_pool.add(name)
        indices_offset = indices_base + len(indices)
        records += struct.pack("<QH6sII", indices_offset, len(look_indices), b"\x00" * 6, name_hash, name_offset)
        if look_indices:
            indices += struct.pack(f"<{len(look_indices)}H", *look_indices)

    return bytes(struct.pack("<B", len(groups)) + records + indices)


def _build_look_blocks(subset_count, string_pool, template, arm=None, subset_index_map=None):
    look_name = "default"
    look_name_offset = string_pool.add(look_name)
    look_hash = string_crc32(look_name)

    original_look = template.payload(BLOCK_HASHES["ModelLook"])
    original_look_built = template.payload(BLOCK_HASHES["ModelLookBuilt"])
    original_subset_block = template.payload(BLOCK_HASHES["ModelSubset"]) if BLOCK_HASHES["ModelSubset"] in template.blocks else b""
    source_subset_count = len(original_subset_block) // MODEL_SUBSET_RECORD_SIZE
    source_look_count = max(1, len(original_look) // MODEL_LOOK_SIZE)
    source_built_look_count = min(source_look_count, len(original_look_built) // MODEL_LOOK_BUILT_SIZE)
    subset_index_map = subset_index_map or {}
    use_custom_looks = bool(arm and arm.get("engine_model_looks_modified", False))
    custom_looks = _json_list_from_idprop(arm, "engine_model_looks_json") if use_custom_looks else []
    custom_groups = _json_list_from_idprop(arm, "engine_model_look_groups_json") if use_custom_looks else []

    def source_look_index(value, fallback=0):
        try:
            index = int(value)
        except Exception:
            index = fallback
        if 0 <= index < source_built_look_count:
            return index
        return 0 if source_built_look_count else -1

    def source_look_header(index):
        index = source_look_index(index)
        if index < 0:
            return None
        offset = index * MODEL_LOOK_BUILT_SIZE
        if offset + MODEL_LOOK_BUILT_SIZE > len(original_look_built):
            return None
        offsets = list(struct.unpack_from("<7Q", original_look_built, offset))
        counts = list(struct.unpack_from("<6H", original_look_built, offset + 56))
        hashes = struct.unpack_from("<3I", original_look_built, offset + 68)
        return offsets, counts, hashes

    def source_look_section_size(index, section):
        header = source_look_header(index)
        if not header:
            return 0
        offsets, counts, _hashes = header
        data_offset = int(offsets[section])
        if data_offset < 0 or data_offset >= len(original_look_built):
            return 0
        if section < 6:
            return max(0, int(counts[section]) * 2)
        candidates = [len(original_look_built)]
        for source_index in range(source_built_look_count):
            other = source_look_header(source_index)
            if not other:
                continue
            for other_offset in other[0]:
                other_offset = int(other_offset)
                if data_offset < other_offset <= len(original_look_built):
                    candidates.append(other_offset)
        return max(0, min(candidates) - data_offset)

    def source_look_section(index, section):
        header = source_look_header(index)
        if not header:
            return b"", 0
        offsets, counts, _hashes = header
        data_offset = int(offsets[section])
        size = source_look_section_size(index, section)
        if size <= 0 or data_offset < 0 or data_offset + size > len(original_look_built):
            return b"", 0
        count = int(counts[section]) if section < 6 else 0
        return bytes(original_look_built[data_offset:data_offset + size]), count

    look_defs = []
    if custom_looks:
        for look_index, look_info in enumerate(custom_looks):
            mapped_ids = _mapped_subset_ids(look_info.get("subset_ids", []), subset_index_map, subset_count, allow_direct=True)
            name = str(look_info.get("name", "") or f"Look {look_index}")
            name_hash = int(look_info.get("name_hash", 0) or 0) & U32_MASK
            if not name_hash:
                name_hash = string_crc32(name)
            look_defs.append({
                "ids": mapped_ids,
                "name": name,
                "hash": name_hash,
                "offset": string_pool.add(name),
                "source_index": source_look_index(look_info.get("index", look_index), look_index),
            })
    if not look_defs:
        for look_index in range(source_look_count):
            original_offset = look_index * MODEL_LOOK_BUILT_SIZE
            mapped_ids = []
            if original_offset + MODEL_LOOK_BUILT_SIZE <= len(original_look_built):
                subset_ids_offset = struct.unpack_from("<Q", original_look_built, original_offset)[0]
                subset_id_count = struct.unpack_from("<H", original_look_built, original_offset + 56)[0]
                original_name_hash, _original_name_hash_lower, original_name_offset = struct.unpack_from(
                    "<3I", original_look_built, original_offset + 68
                )
                ids_start = int(subset_ids_offset)
                ids_end = ids_start + int(subset_id_count) * 2
                if 0 <= ids_start <= ids_end <= len(original_look_built) and subset_id_count:
                    source_ids = list(struct.unpack_from(f"<{int(subset_id_count)}H", original_look_built, ids_start))
                    mapped_ids = _mapped_subset_ids(source_ids, subset_index_map, subset_count)
            else:
                original_name_hash = look_hash
                original_name_offset = look_name_offset
            if not mapped_ids and look_index == 0:
                mapped_ids = [index for index in range(min(source_subset_count, subset_count))]
            look_defs.append({
                "ids": mapped_ids,
                "name": f"Look {look_index}",
                "hash": original_name_hash,
                "offset": original_name_offset,
                "source_index": source_look_index(look_index, look_index),
            })
    look_count = max(1, len(look_defs))

    look = bytearray()
    for look_index, look_def in enumerate(look_defs):
        ids = look_def["ids"]
        source_index = int(look_def.get("source_index", look_index))
        source_offset = source_index * MODEL_LOOK_SIZE
        if 0 <= source_offset and source_offset + MODEL_LOOK_SIZE <= len(original_look):
            record = bytearray(original_look[source_offset:source_offset + MODEL_LOOK_SIZE])
        else:
            record = bytearray(b"\x00" * MODEL_LOOK_SIZE)
        for lod_index in range(8):
            struct.pack_into("<HH", record, lod_index * 4, 0, len(ids))
        look += record

    look_built_headers = bytearray()
    headers_size = look_count * MODEL_LOOK_BUILT_SIZE
    look_built_data = bytearray()
    for look_index, look_def in enumerate(look_defs):
        ids = look_def["ids"]
        source_index = int(look_def.get("source_index", look_index))
        section_offsets = []
        section_counts = []

        section_offsets.append(headers_size + len(look_built_data))
        section_counts.append(len(ids))
        if ids:
            look_built_data += struct.pack(f"<{len(ids)}H", *ids)

        for section in range(1, 7):
            chunk, count = source_look_section(source_index, section)
            section_offsets.append(headers_size + len(look_built_data))
            if section < 6:
                section_counts.append(count)
            if chunk:
                look_built_data += chunk

        name_hash = int(look_def.get("hash", look_hash)) & U32_MASK
        name_offset = int(look_def.get("offset", look_name_offset))
        look_built_headers += struct.pack(
            "<7Q6H3I",
            section_offsets[0],
            section_offsets[1],
            section_offsets[2],
            section_offsets[3],
            section_offsets[4],
            section_offsets[5],
            section_offsets[6],
            section_counts[0],
            section_counts[1],
            section_counts[2],
            section_counts[3],
            section_counts[4],
            section_counts[5],
            name_hash,
            name_hash,
            name_offset,
        )

    if custom_looks:
        look_group = _build_look_group_block(custom_groups, look_count, string_pool)
    else:
        look_group = template.payload(BLOCK_HASHES["ModelLookGroup"])
    if not look_group:
        look_group = bytearray()
        look_group += struct.pack("<B", 1)
        look_group += struct.pack("<QH6sII", MODEL_LOOK_GROUP_SIZE, look_count, b"\x00" * 6, look_hash, look_name_offset)
        look_group += struct.pack(f"<{look_count}H", *range(look_count))
    return bytes(look), bytes(look_built_headers + look_built_data), bytes(look_group)


def _valid_model_bounds(bsphere, aabb):
    return (
        bsphere is not None
        and aabb is not None
        and len(bsphere) == 4
        and len(aabb) == 3
        and all(math.isfinite(float(v)) for v in tuple(bsphere) + tuple(aabb))
        and float(bsphere[3]) > 0.0
        and all(float(v) >= 0.0 for v in aabb)
    )


def _float_list_from_json_prop(owner, key, count):
    if owner is None:
        return None
    raw = owner.get(key)
    if raw is None:
        return None
    try:
        values = json.loads(raw) if isinstance(raw, str) else list(raw)
    except Exception:
        return None
    if len(values) != count:
        return None
    try:
        values = tuple(float(v) for v in values)
    except Exception:
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return values


def _positive_float_prop(owner, key):
    if owner is None:
        return None
    try:
        value = float(owner.get(key))
    except Exception:
        return None
    if math.isfinite(value) and value > 0.0:
        return value
    return None


def _source_model_built_state(model_built, arm=None):
    bsphere = _float_list_from_json_prop(arm, "engine_model_source_bsphere_json", 4)
    aabb = _float_list_from_json_prop(arm, "engine_model_source_aabb_json", 3)
    source_bounds = (bsphere[:3], aabb, bsphere[3]) if _valid_model_bounds(bsphere, aabb) else None
    source_common_mpu = _positive_float_prop(arm, "engine_model_source_common_mpu")
    source_vertex_mpu = _positive_float_prop(arm, "engine_model_source_vertex_mpu")

    try:
        if source_bounds is None:
            block_bsphere = struct.unpack_from("<4f", model_built, MODEL_BUILT_BSPHERE_OFFSET)
            block_aabb = struct.unpack_from("<3f", model_built, MODEL_BUILT_AABB_EXTENTS_OFFSET)
            if _valid_model_bounds(block_bsphere, block_aabb):
                source_bounds = (block_bsphere[:3], block_aabb, block_bsphere[3])
        if source_common_mpu is None:
            value = struct.unpack_from("<f", model_built, MODEL_BUILT_COMMON_MPU_OFFSET)[0]
            if math.isfinite(value) and value > 0.0:
                source_common_mpu = float(value)
        if source_vertex_mpu is None:
            value = struct.unpack_from("<f", model_built, MODEL_BUILT_VERTEX_MPU_OFFSET)[0]
            if math.isfinite(value) and value > 0.0:
                source_vertex_mpu = float(value)
    except Exception:
        pass

    return source_bounds, source_common_mpu, source_vertex_mpu


def _higher_power_of_two(value):
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _model_mpu_from_bounds(center, extents):
    mins = [float(center[axis]) - float(extents[axis]) for axis in range(3)]
    maxs = [float(center[axis]) + float(extents[axis]) for axis in range(3)]
    min_component = min(mins)
    max_component = max(maxs)

    unbounded_int_min = int(math.ceil(abs(min_component) * float(1 << 15)))
    unbounded_int_max = int(math.ceil(abs(max_component) * float(1 << 15)))
    unbounded_int_range = max(unbounded_int_min, unbounded_int_max + 1)
    max_component_aligned = float(_higher_power_of_two(unbounded_int_range) >> 15)

    vertex_range = max(max_component_aligned, 8.0)
    return vertex_range / float(1 << 15)


def _combine_model_bounds(source_bounds, subset_stats):
    bounds = []
    radius_sources = []
    if source_bounds:
        source_center, source_extents, source_radius = source_bounds
        source_center = tuple(float(v) for v in source_center)
        source_extents = tuple(float(v) for v in source_extents)
        source_radius = float(source_radius)
        source_mins = tuple(source_center[axis] - source_extents[axis] for axis in range(3))
        source_maxs = tuple(source_center[axis] + source_extents[axis] for axis in range(3))
        if subset_stats:
            all_inside_source = True
            for stat in subset_stats:
                center = tuple(float(stat["center"][axis]) for axis in range(3))
                extents = tuple(float(stat["extents"][axis]) for axis in range(3))
                radius = float(stat.get("radius", 0.0))
                if any(center[axis] - extents[axis] < source_mins[axis] for axis in range(3)):
                    all_inside_source = False
                    break
                if any(center[axis] + extents[axis] > source_maxs[axis] for axis in range(3)):
                    all_inside_source = False
                    break
                if _vec_len(_vec_sub(center, source_center)) + radius > source_radius:
                    all_inside_source = False
                    break
            if all_inside_source:
                return source_center, source_extents, source_radius
        bounds.append((
            source_mins,
            source_maxs,
        ))
        radius_sources.append((source_center, source_radius))
    for stat in subset_stats:
        center = tuple(float(stat["center"][axis]) for axis in range(3))
        extents = tuple(float(stat["extents"][axis]) for axis in range(3))
        radius = float(stat.get("radius", 0.0))
        bounds.append((
            tuple(center[axis] - extents[axis] for axis in range(3)),
            tuple(center[axis] + extents[axis] for axis in range(3)),
        ))
        radius_sources.append((center, radius))

    if not bounds:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0

    mins = [min(item[0][axis] for item in bounds) for axis in range(3)]
    maxs = [max(item[1][axis] for item in bounds) for axis in range(3)]
    center = tuple((mins[axis] + maxs[axis]) * 0.5 for axis in range(3))
    extents = tuple((maxs[axis] - mins[axis]) * 0.5 for axis in range(3))
    radius_padding = max(MODEL_GLOBAL_BOUNDS_PADDING, max((radius for _center, radius in radius_sources), default=0.0) * 0.01)
    radius = max((_vec_len(_vec_sub(item_center, center)) + item_radius for item_center, item_radius in radius_sources), default=0.0)
    return center, tuple(value + radius_padding for value in extents), radius + radius_padding


def _build_model_built_block(template, subset_stats, arm=None, has_morph=False, has_smooth=False):
    block_hash = BLOCK_HASHES["ModelBuilt"]
    original = bytearray(template.payload(block_hash) if block_hash in template.blocks else b"\x00" * MODEL_BUILT_SIZE)
    if len(original) < MODEL_BUILT_SIZE:
        original += b"\x00" * (MODEL_BUILT_SIZE - len(original))
    model_built = original[:MODEL_BUILT_SIZE]

    flags = struct.unpack_from("<Q", model_built, MODEL_BUILT_FLAGS_OFFSET)[0]
    flags &= ~(MODEL_FLAG_ANIM_VERT | MODEL_FLAG_ANIM_DYNAMICS | MODEL_FLAG_USES_AUTO_LODS)
    if has_morph:
        flags |= MODEL_FLAG_ANIM_VERT
    if not any(stat.get("skinned") for stat in subset_stats):
        flags &= ~(MODEL_FLAG_HAS_SKINNING | MODEL_FLAG_HAS_GPU_SKINNING)
    struct.pack_into("<Q", model_built, MODEL_BUILT_FLAGS_OFFSET, flags)

    content_flags = struct.unpack_from("<H", model_built, MODEL_BUILT_CONTENT_FLAGS_OFFSET)[0]
    content_flags &= ~(
        CONTENT_FLAG_ANIM_MORPH
        | CONTENT_FLAG_ANIM_ZIVA
        | CONTENT_FLAG_ANIM_VERT_SMOOTH
        | CONTENT_FLAG_USES_AUTO_LODS
    )
    if has_morph:
        content_flags |= CONTENT_FLAG_ANIM_MORPH
    if has_smooth:
        content_flags |= CONTENT_FLAG_ANIM_VERT_SMOOTH
    struct.pack_into("<H", model_built, MODEL_BUILT_CONTENT_FLAGS_OFFSET, content_flags)

    source_bounds, source_common_mpu, source_vertex_mpu = _source_model_built_state(model_built, arm=arm)

    center, extents, radius = _combine_model_bounds(source_bounds, subset_stats)

    common_mpu = source_common_mpu
    bounds_mpu = _model_mpu_from_bounds(center, extents) if radius > 0.0 else MODEL_DEFAULT_MPU
    if common_mpu is None:
        common_mpu = bounds_mpu
    else:
        common_mpu = max(common_mpu, bounds_mpu)
    if not (math.isfinite(common_mpu) and common_mpu > 0.0):
        common_mpu = MODEL_DEFAULT_MPU

    subset_vertex_mpus = []
    for stat in subset_stats:
        try:
            stat_mpu = float(stat.get("mpu", 0.0) or 0.0)
        except Exception:
            continue
        if math.isfinite(stat_mpu) and stat_mpu > 0.0:
            subset_vertex_mpus.append(stat_mpu)
    subset_vertex_mpu = max(subset_vertex_mpus) if subset_vertex_mpus else None
    vertex_mpu = subset_vertex_mpu if subset_vertex_mpu is not None else source_vertex_mpu
    if vertex_mpu is None:
        vertex_mpu = common_mpu
    if not (math.isfinite(vertex_mpu) and vertex_mpu > 0.0):
        vertex_mpu = common_mpu

    struct.pack_into("<H", model_built, MODEL_BUILT_FADE_OUT_DIST_OFFSET, 0)
    struct.pack_into("<4f", model_built, MODEL_BUILT_BSPHERE_OFFSET, center[0], center[1], center[2], radius)
    struct.pack_into("<3f", model_built, MODEL_BUILT_AABB_EXTENTS_OFFSET, extents[0], extents[1], extents[2])
    struct.pack_into("<f", model_built, MODEL_BUILT_COMMON_MPU_OFFSET, common_mpu)
    struct.pack_into("<f", model_built, MODEL_BUILT_VERTEX_MPU_OFFSET, vertex_mpu)
    struct.pack_into("<I", model_built, MODEL_BUILT_CUSTOM_STREAM_COUNT_OFFSET, sum(stat["custom_stream_count"] for stat in subset_stats))
    struct.pack_into("<H", model_built, MODEL_BUILT_SUBSET_LOD_MASK_COUNT_OFFSET, len(subset_stats))
    struct.pack_into("<b", model_built, MODEL_BUILT_STRAND_SUBSET_COUNT_OFFSET, 0)
    return bytes(model_built)


def _source_morph_targets_for_older_scene(template, mesh_objects):
    decoded = decode_model_morph2(template.data, template.blocks)
    if not decoded:
        return {}, 0
    objects_by_subset = {}
    for obj in mesh_objects:
        try:
            subset_index = int(obj.get("engine_subset_index", -1))
        except Exception:
            continue
        if subset_index >= 0:
            objects_by_subset.setdefault(subset_index, []).append(obj)

    source_subset = template.payload(BLOCK_HASHES["ModelSubset"])
    geom_base, _geom_size = template.blocks[BLOCK_HASHES["ModelSubsetGeomData"]]
    source_subset_count = len(source_subset) // MODEL_SUBSET_RECORD_SIZE
    result = {}
    skipped_records = 0
    for target in decoded.get("targets", []):
        for target_subset in target.get("subsets", []):
            subset_index = int(target_subset["subset_index"])
            objects = objects_by_subset.get(subset_index, [])
            if not objects:
                skipped_records += 1
                continue
            if len(objects) != 1 or not 0 <= subset_index < source_subset_count:
                raise ValueError(
                    f"The saved facial shape {target['name']!r} cannot be matched to one mesh part. "
                    "Re-import the original model with Import Shape Keys enabled, then repeat your edits."
                )
            obj = objects[0]
            mesh = obj.data
            record_offset = subset_index * MODEL_SUBSET_RECORD_SIZE
            source_index_count = struct.unpack_from("<I", source_subset, record_offset + MODEL_SUBSET_INDEX_COUNT_OFFSET)[0]
            source_vertex_count = struct.unpack_from("<I", source_subset, record_offset + MODEL_SUBSET_VERTEX_COUNT_OFFSET)[0]
            if len(mesh.vertices) != source_vertex_count:
                raise ValueError(
                    f"{obj.name}'s vertex count changed, so the saved facial shape {target['name']!r} no "
                    "longer fits. Re-import the original model with Import Shape Keys enabled before changing "
                    "the mesh topology."
                )
            subset_base = struct.unpack_from("<I", source_subset, record_offset + MODEL_SUBSET_BASE_OFFSET)[0]
            index_offset = struct.unpack_from("<I", source_subset, record_offset + MODEL_SUBSET_INDEX_DATA_OFFSET)[0]
            source_indices = list(struct.unpack_from(
                f"<{source_index_count}H",
                template.data,
                geom_base + int(subset_base) + int(index_offset),
            ))
            source_triangles = sorted(
                tuple(sorted(source_indices[index:index + 3]))
                for index in range(0, len(source_indices), 3)
                if len(source_indices[index:index + 3]) == 3
            )
            mesh.calc_loop_triangles()
            current_triangles = sorted(
                tuple(sorted(int(mesh.loops[loop_index].vertex_index) for loop_index in triangle.loops))
                for triangle in mesh.loop_triangles
            )
            if current_triangles != source_triangles:
                raise ValueError(
                    f"{obj.name}'s faces changed, so the saved facial shape {target['name']!r} no longer fits. "
                    "Re-import the original model with Import Shape Keys enabled before changing the mesh faces."
                )
            deltas = {int(index): tuple(value) for index, value in target_subset.get("deltas", {}).items()}
            if any(index < 0 or index >= source_vertex_count for index in deltas):
                raise ValueError(
                    f"The original file has damaged facial-shape data for {target['name']!r}. "
                    "Try a fresh copy of the original extracted .model file."
                )
            result.setdefault(subset_index, []).append({
                "name": str(target["name"]),
                "hash": int(target["hash"]) & U32_MASK,
                "source_index": int(target.get("index", -1)),
                "deltas": deltas,
            })
    if not result:
        raise ValueError(
            "The facial shapes from this older Blender file cannot be matched to the loaded meshes. "
            "Re-import the original model with Import Shape Keys enabled."
        )
    return result, skipped_records



def _build_inert_look_bvh_blocks(template):
    replacements = {}
    bvh_hash = BLOCK_HASHES.get("ModelLookBVHInfo")
    if bvh_hash in template.blocks:
        replacements[bvh_hash] = struct.pack("<4I", 0, 0, 0, 0)
    bvh_lod_hash = BLOCK_HASHES.get("ModelLookBVHLoDInfo")
    if bvh_lod_hash in template.blocks:
        replacements[bvh_lod_hash] = b"\x00" * 64
    return replacements


def _block_alignment(block_hash):
    cacheline_blocks = {
        BLOCK_HASHES.get("ModelSubset"),
        BLOCK_HASHES.get("ModelLook"),
        BLOCK_HASHES.get("ModelLookBuilt"),
        BLOCK_HASHES.get("ModelLookBVHInfo"),
    }
    return DAT1_CACHELINE_ALIGN if block_hash in cacheline_blocks else DAT1_BLOCK_ALIGN


def _relocate_model_string_offsets(block_hash, payload, delta):
    if not delta or not payload:
        return payload
    data = bytearray(payload)
    if block_hash == BLOCK_HASHES["ModelMaterial"]:
        material_count = len(data) // (MODEL_MATERIAL_INFO_SIZE + MODEL_MATERIAL_SIZE)
        for index in range(material_count):
            base = index * MODEL_MATERIAL_INFO_SIZE
            for field_offset in (0, 8):
                value = struct.unpack_from("<I", data, base + field_offset)[0]
                if value:
                    struct.pack_into("<I", data, base + field_offset, value + delta)
    elif block_hash == BLOCK_HASHES["ModelLookBuilt"]:
        # Headers occupy the leading fixed-size record array
        look_count = 0
        if len(data) >= MODEL_LOOK_BUILT_SIZE:
            first_section = struct.unpack_from("<Q", data, 0)[0]
            if first_section % MODEL_LOOK_BUILT_SIZE == 0:
                look_count = min(len(data) // MODEL_LOOK_BUILT_SIZE, int(first_section // MODEL_LOOK_BUILT_SIZE))
        for index in range(look_count):
            field_offset = index * MODEL_LOOK_BUILT_SIZE + 76
            value = struct.unpack_from("<I", data, field_offset)[0]
            if value:
                struct.pack_into("<I", data, field_offset, value + delta)
    elif block_hash == BLOCK_HASHES["ModelLookGroup"]:
        group_count = int(data[0]) if data else 0
        for index in range(group_count):
            field_offset = 1 + index * MODEL_LOOK_GROUP_SIZE + 20
            if field_offset + 4 <= len(data):
                value = struct.unpack_from("<I", data, field_offset)[0]
                if value:
                    struct.pack_into("<I", data, field_offset, value + delta)
    return bytes(data)


def _rebuild_dat1(template, replacements, string_pool, remove_hashes=None):
    if template.fixup_count != 0:
        raise ValueError(
            "This particular game-model layout is not supported yet. Try a different original .model file. "
            "If you need this exact model, share the System Console error with the add-on developer."
        )

    remove_hashes = set(remove_hashes or ())
    physical_hashes = [
        entry[0]
        for entry in sorted(template.entries, key=lambda item: item[1])
        if entry[0] not in remove_hashes
    ]
    geom_hash = BLOCK_HASHES["ModelSubsetGeomData"]
    added_hashes = [
        block_hash for block_hash in replacements
        if block_hash not in physical_hashes and block_hash not in remove_hashes and block_hash != geom_hash
    ]
    if geom_hash in physical_hashes:
        physical_hashes = [h for h in physical_hashes if h != geom_hash] + sorted(added_hashes) + [geom_hash]
    else:
        physical_hashes.extend(sorted(added_hashes))

    strings = string_pool.bytes()
    table_end = DAT1_HEADER_SIZE + len(physical_hashes) * DAT1_BLOCK_TABLE_ENTRY_SIZE + len(template.fixup_table)
    string_base = max(template.sb_offset, table_end)
    string_delta = string_base - template.sb_offset
    cursor = string_base + len(strings)
    payload_by_hash = {}
    offset_by_hash = {}
    body = bytearray()

    for block_hash in physical_hashes:
        payload = replacements.get(block_hash)
        if payload is None:
            payload = template.payload(block_hash)
        if string_delta:
            payload = _relocate_model_string_offsets(block_hash, payload, string_delta)
        if block_hash == geom_hash:
            original_geom_offset, original_geom_size = template.blocks[geom_hash]
            if len(payload) < original_geom_size:
                payload += b"\x00" * (original_geom_size - len(payload))
        alignment = _block_alignment(block_hash)
        aligned = _align(cursor, alignment)
        if block_hash == geom_hash and aligned <= original_geom_offset:
            aligned = original_geom_offset
        pad = aligned - cursor
        if pad:
            body += b"\x00" * pad
            cursor = aligned
        offset_by_hash[block_hash] = cursor
        payload_by_hash[block_hash] = payload
        body += payload
        cursor += len(payload)

    block_table_entries = []
    for block_hash in sorted(payload_by_hash):
        payload = payload_by_hash[block_hash]
        block_table_entries.append(struct.pack("<III", block_hash, offset_by_hash[block_hash], len(payload)))
    block_table = b"".join(block_table_entries)
    # Preserve the source string buffer's absolute DAT1 offset. Many model
    # records store absolute string pointers, including records we otherwise
    # byte-preserve. Removing a block-table entry must therefore leave padding
    # instead of sliding every later payload toward the header.
    header_padding = b"\x00" * max(0, string_base - table_end)
    declared_size = DAT1_HEADER_SIZE + len(block_table) + len(template.fixup_table) + len(header_padding) + len(strings) + len(body)
    header = struct.pack("<IIIHH", DAT1_FILE_ID, template.version, declared_size, len(payload_by_hash), template.fixup_count)
    out = header + block_table + template.fixup_table + header_padding + strings + bytes(body)
    return out, offset_by_hash.get(geom_hash, 0), len(payload_by_hash.get(geom_hash, b""))


def _asset_chunk_info(size):
    size = int(size)
    if size < 0 or size > ASSET_CHUNK_UNCOMPRESSED_MASK:
        raise ValueError(
            "The exported model is too large for one game file. Split large meshes into smaller objects, "
            "then export again."
        )
    return (
        size
        | (size << ASSET_CHUNK_COMPRESSED_SHIFT)
        | (ASSET_COMPRESSION_NONE << ASSET_CHUNK_COMPRESSION_SHIFT)
    )


def _build_stg_header(version, topology_size, bulk_size):
    chunks = [_asset_chunk_info(topology_size)]
    if bulk_size > 0:
        chunks.append(_asset_chunk_info(bulk_size))
    serialized_header = struct.pack("<IBBH", version, 0, len(chunks), 0)
    serialized_header += b"".join(struct.pack("<Q", chunk) for chunk in chunks)
    stg = bytearray()
    stg += struct.pack("<IIII", STG_MAGIC, STG_VERSION, len(serialized_header), 0)
    stg += serialized_header
    _align_buffer(stg, STG_HEADER_ALIGN)
    return bytes(stg)


def _expected_original_model_name(arm):
    source_name = os.path.basename(_source_path_from_armature(arm))
    if source_name:
        return source_name

    arm_name = str(getattr(arm, "name", "") or "skeleton")
    model_name = re.match(r"^(.*\.model)(?:\.\d{3})?$", arm_name, flags=re.IGNORECASE)
    if model_name:
        return model_name.group(1)
    return f"{arm_name}.model"


class MODEL_OT_select_original_model_for_export(Operator, ImportHelper):
    bl_idname = "model.select_original_model_for_export"
    bl_label = "Please Select Original .model"
    bl_description = "Select the original skeleton model to use as the injection template"
    filename_ext = ".model"
    filter_glob: StringProperty(default="*.model;*.dat1", options={'HIDDEN'})
    stg_mode: StringProperty(default="SCENE", options={'HIDDEN', 'SKIP_SAVE'})
    expected_model_name: StringProperty(options={'HIDDEN', 'SKIP_SAVE'})

    def draw(self, context):
        expected_name = str(self.expected_model_name or "skeleton.model")
        self.layout.label(text=f"Please select original {expected_name}", icon='ARMATURE_DATA')

    def invoke(self, context, event):
        arm = _resolve_model_armature(context)
        if not arm:
            self.report(
                {'ERROR'},
                "Select the model skeleton (Armature) or one of its meshes, then click Export again.",
            )
            return {'CANCELLED'}

        expected_name = str(self.expected_model_name or _expected_original_model_name(arm))
        self.expected_model_name = expected_name
        missing_path = _source_path_from_armature(arm)
        if missing_path:
            self.filepath = missing_path
        elif not self.filepath:
            self.filepath = expected_name
        self.report(
            {'INFO'},
            f"Please select original {expected_name}.",
        )
        return ImportHelper.invoke(self, context, event)

    def execute(self, context):
        arm = _resolve_model_armature(context)
        if not arm:
            self.report({'ERROR'}, "The selected model skeleton is no longer available.")
            return {'CANCELLED'}

        source_path = os.path.abspath(self.filepath)
        if not os.path.isfile(source_path):
            self.report({'ERROR'}, "Please select an existing original skeleton .model file.")
            return {'CANCELLED'}

        try:
            template = _Dat1Template(source_path)
        except Exception:
            log_exception("Could not read replacement source model template %s", source_path)
            self.report(
                {'ERROR'},
                "That file is not a readable original game .model. Please select another skeleton model.",
            )
            return {'CANCELLED'}

        arm["engine_model_source_path"] = source_path
        arm["engine_model_source_had_stg"] = bool(template.had_stg)
        source_has_morphs = BLOCK_HASHES["ModelAnimMorph2Info"] in template.blocks
        source_has_ziva = BLOCK_HASHES["ModelAnimZiva2Info"] in template.blocks
        arm["engine_model_source_has_morphs"] = source_has_morphs
        arm["engine_model_source_has_ziva"] = source_has_ziva
        if source_has_morphs:
            try:
                source_morph = decode_model_morph2(template.data, template.blocks)
                arm["engine_model_morph_target_count"] = int(
                    len((source_morph or {}).get("targets", []))
                )
            except Exception:
                log_exception("Could not count source model shape keys %s", source_path)
        self.report({'INFO'}, f"Using {os.path.basename(source_path)} as the injection model.")
        return bpy.ops.export_scene.engine_model(
            'INVOKE_DEFAULT',
            stg_mode=str(self.stg_mode or "SCENE"),
        )


class ExportEngineModel(Operator, ExportHelper):
    bl_idname = "export_scene.engine_model"
    bl_label = "Export Luna Engine Model"
    bl_description = "Export selected/imported Luna Engine Model geometry using the original .model as a template"
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}
    filename_ext = ".model"
    filter_glob: StringProperty(default="*.model;*.dat1", options={'HIDDEN'})
    stg_mode: EnumProperty(
        name="Output Format",
        items=[
            ("SCENE", "Scene Setting", "Use the Luna Engine panel STG checkbox"),
            ("AUTO", "Auto", "Match the imported source file wrapper"),
            ("STG", "STG+DAT1", "Write a model-style STG header before DAT1"),
            ("RAW", "Raw DAT1", "Write DAT1 without an STG header"),
        ],
        default="SCENE",
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    def draw(self, context):
        return

    def invoke(self, context, event):
        arm = _resolve_model_armature(context)
        if not arm:
            self.report(
                {'ERROR'},
                "Nothing from an imported model is selected. Select the model skeleton (Armature) or one of "
                "its meshes, then click Export again.",
            )
            return {'CANCELLED'}

        source_path = _source_path_from_armature(arm)
        if not source_path or not os.path.isfile(source_path):
            return bpy.ops.model.select_original_model_for_export(
                'INVOKE_DEFAULT',
                stg_mode=str(self.stg_mode or "SCENE"),
                expected_model_name=_expected_original_model_name(arm),
            )
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        arm = _resolve_model_armature(context)
        if not arm:
            self.report(
                {'ERROR'},
                "Nothing from an imported model is selected. Select the model skeleton (Armature) or one of "
                "its meshes, then click Export again.",
            )
            return {'CANCELLED'}

        source_path = _source_path_from_armature(arm)
        if not source_path or not os.path.isfile(source_path):
            self.report(
                {'ERROR'},
                "I can't find the original .model file used by this Blender scene. Import the original model "
                "again, then export.",
            )
            return {'CANCELLED'}

        try:
            template = _Dat1Template(source_path)
        except Exception as exc:
            log_exception("Could not read source model template %s", source_path)
            self.report(
                {'ERROR'},
                "I couldn't open the original .model file. Make sure it still exists and is an original "
                "extracted game model, then import it again.",
            )
            return {'CANCELLED'}

        source_has_morph = BLOCK_HASHES["ModelAnimMorph2Info"] in template.blocks
        source_has_ziva = BLOCK_HASHES["ModelAnimZiva2Info"] in template.blocks
        source_has_smooth = BLOCK_HASHES["ModelAnimVertSmoothInfo"] in template.blocks
        discard_unimported_morphs = bool(getattr(arm, "engine_model_discard_unimported_morphs", False))
        recover_source_morph = (
            source_has_morph
            and not bool(arm.get("engine_model_shape_keys_imported", False))
            and not discard_unimported_morphs
        )

        required = ("ModelBuilt", "ModelMaterial", "ModelLook", "ModelLookGroup", "ModelLookBuilt", "ModelSubset", "ModelSubsetGeomData")
        missing = [name for name in required if BLOCK_HASHES[name] not in template.blocks]
        if missing:
            self.report(
                {'ERROR'},
                "The chosen source file is not a complete game model. Import a different original .model "
                f"file and try again. Missing internal data: {', '.join(missing)}.",
            )
            return {'CANCELLED'}

        mesh_objects = [
            obj for obj in bpy.data.objects
            if getattr(obj, "type", None) == "MESH" and getattr(obj, "parent", None) == arm
            and obj.get("engine_bounds_type", "") != "subset_aabb"
        ]
        resolve_subset_index_collisions(arm)
        sanitize_model_look_metadata(arm, mark_modified=True)

        def subset_sort_key(obj):
            value = obj.get("engine_subset_index", None)
            return (999999 if value is None else int(value), obj.name)

        mesh_objects.sort(key=subset_sort_key)
        if not mesh_objects:
            self.report(
                {'ERROR'},
                "No model meshes were found under the selected skeleton. Parent at least one mesh directly "
                "to the Armature, then export again.",
            )
            return {'CANCELLED'}

        wm = context.window_manager
        wm.progress_begin(0, 100)
        export_warnings = []
        try:
            wm.progress_update(10)
            original_materials = _parse_model_materials(template.data, template.blocks)
            material_entries, material_indices = _build_material_entries(mesh_objects, original_materials, export_warnings)
            string_pool = _StringPool(template.sb_offset, template.string_buffer)

            wm.progress_update(30)
            hierarchy_hash = BLOCK_HASHES["ModelJointHierarchy"]
            if hierarchy_hash in template.blocks:
                hierarchy = template.payload(hierarchy_hash)
                source_joint_count = struct.unpack_from("<H", hierarchy, 2)[0] if len(hierarchy) >= 4 else None
            else:
                source_joint_count = 0
            source_morph_targets_by_subset = {}
            if recover_source_morph:
                try:
                    source_morph_targets_by_subset, skipped_morph_subset_records = _source_morph_targets_for_older_scene(
                        template,
                        mesh_objects,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{exc} If you do not need facial animation, turn on 'Discard Unimported Morph2' "
                        "under Model > Export and try again."
                    ) from exc
                log_debug(
                    "Older scene had no imported shape keys; compatible source facial shapes were recovered."
                )
                if skipped_morph_subset_records:
                    export_warnings.append(
                        f"{skipped_morph_subset_records} facial-shape part(s) could not be matched because their "
                        "meshes are not loaded. Re-import with Import All LODs and Import Shape Keys enabled if "
                        "you need those parts."
                    )
            elif source_has_morph and discard_unimported_morphs:
                export_warnings.append(
                    "Facial animation was left out because 'Discard Unimported Morph2' is turned on. Turn it "
                    "off and re-import with Import Shape Keys enabled if you want facial animation."
                )

            subset_block, geom_block, subset_stats, subset_index_map, morph_targets = _build_geometry_and_subset_blocks(
                mesh_objects,
                arm,
                material_indices,
                template,
                source_joint_count,
                export_warnings=export_warnings,
                source_morph_targets_by_subset=source_morph_targets_by_subset,
            )
            morph_info_block, morph_geom_suffix, morph_metadata = encode_model_morph2(
                morph_targets,
                len(geom_block),
            )
            has_morph = morph_info_block is not None
            if has_morph:
                geom_block += morph_geom_suffix
            material_block = _build_material_block(material_entries, string_pool)
            generated_subset_count = len(subset_stats)
            look_block, look_built_block, look_group_block = _build_look_blocks(
                generated_subset_count,
                string_pool,
                template,
                arm=arm,
                subset_index_map=subset_index_map,
            )
# i dont think this works
            has_smooth = bool(has_morph and source_has_smooth and not source_has_ziva)
            model_built_block = _build_model_built_block(
                template,
                subset_stats,
                arm=arm,
                has_morph=has_morph,
                has_smooth=has_smooth,
            )

            replacements = {
                BLOCK_HASHES["ModelBuilt"]: model_built_block,
                BLOCK_HASHES["ModelMaterial"]: material_block,
                BLOCK_HASHES["ModelLook"]: look_block,
                BLOCK_HASHES["ModelLookBuilt"]: look_built_block,
                BLOCK_HASHES["ModelLookGroup"]: look_group_block,
                BLOCK_HASHES["ModelSubset"]: subset_block,
                BLOCK_HASHES["ModelSubsetGeomData"]: geom_block,
            }
            remove_hashes = set()
            if has_morph:
                replacements[BLOCK_HASHES["ModelAnimVertInfo2"]] = b"\x00" * 40
                replacements[BLOCK_HASHES["ModelAnimMorph2Info"]] = morph_info_block
                if has_smooth:
                    smooth_info_block, smooth_metadata = encode_model_smooth2(subset_stats)
                    replacements[BLOCK_HASHES["ModelAnimVertSmoothInfo"]] = smooth_info_block
                else:
                    remove_hashes.add(BLOCK_HASHES["ModelAnimVertSmoothInfo"])
                if source_has_ziva:
                    remove_hashes.add(BLOCK_HASHES["ModelAnimZiva2Info"])
                    log_debug(
                        "Converted registered shape keys from Ziva to Morph2 while preserving source shading."
                    )
            elif source_has_morph:
                remove_hashes.update({
                    BLOCK_HASHES["ModelAnimMorph2Info"],
                    BLOCK_HASHES["ModelAnimVertSmoothInfo"],
                    BLOCK_HASHES["ModelAnimVertInfo2"],
                })
            elif source_has_ziva:
                remove_hashes.update({
                    BLOCK_HASHES["ModelAnimZiva2Info"],
                    BLOCK_HASHES["ModelAnimVertSmoothInfo"],
                    BLOCK_HASHES["ModelAnimVertInfo2"],
                })
            replacements.update(_build_inert_look_bvh_blocks(template))

            wm.progress_update(75)
            dat1, geom_offset, geom_size = _rebuild_dat1(
                template,
                replacements,
                string_pool,
                remove_hashes=remove_hashes,
            )
            source_had_stg = bool(template.had_stg or arm.get("engine_model_source_had_stg", False))
            stg_mode = str(getattr(self, "stg_mode", "SCENE") or "SCENE")
            if stg_mode == "STG":
                add_stg = True
            elif stg_mode == "RAW":
                add_stg = False
            elif stg_mode == "AUTO":
                add_stg = source_had_stg
            else:
                add_stg = bool(getattr(context.scene, "engine_export_add_stg_header", True))
            if add_stg:
                stg_header = _build_stg_header(template.version, geom_offset, geom_size)
                out = stg_header + dat1
                format_name = "STG+DAT1"
            else:
                out = dat1
                format_name = "raw DAT1"

            with open(self.filepath, "wb") as f:
                f.write(out)
        except Exception as exc:
            log_exception("Model export failed")
            wm.progress_end()
            self.report({'ERROR'}, f"Export couldn't finish. {_friendly_export_error(exc)}")
            return {'CANCELLED'}

        wm.progress_update(100)
        wm.progress_end()
        if export_warnings:
            for warning in export_warnings:
                log_warning("Model export check: %s", warning)
            self.report({'WARNING'}, _format_export_warnings(export_warnings))
        self.report(
            {'INFO'},
            f"Export finished ({format_name}): {len(mesh_objects)} mesh part(s), "
            f"{sum(s['vertex_count'] for s in subset_stats)} vertices, "
            f"{sum(s['index_count'] // 3 for s in subset_stats)} triangles, and "
            f"{len(morph_metadata.get('targets', [])) if has_morph else 0} facial shape(s)."
        )
        return {'FINISHED'}
