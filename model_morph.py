
from .utils import *
from .hashes import BLOCK_HASHES, string_crc32


MORPH_INFO_SIZE = 112
MORPH_TARGET_SIZE = 24
MORPH_SUBSET_SIZE = 8
MORPH_BATCH_SIZE = 20
MORPH_BATCH_DELTA_COUNT = 64
MORPH_FLAG_DEBUG_UNCOMPRESSED = 0x1000
MORPH_DELTA_PRECISION = 0.0001
MORPH_TARGET_MAX = 4096
MORPH_TARGET_SUBSET_MAX = 255
MORPH_NAME_BYTES_MAX = 127
MORPH_STITCH_INDEX_MAX = 32
# this is stupid
SMOOTH_INFO_SIZE = 64
SMOOTH_SUBSET_SIZE = 12
SMOOTH_POSITION_TOLERANCE = 0.0001
SMOOTH_NORMAL_DOT_TOLERANCE = 0.998
MORPH_CONTROL_PREFIX = "engine_morph_"


def _align_morph(value, alignment):
    return (int(value) + int(alignment) - 1) & ~(int(alignment) - 1)


def _object_morph_metadata(obj):
    try:
        metadata = json.loads(str(obj.get("engine_morph_targets_json", "{}") or "{}"))
    except Exception:
        metadata = {}
    return metadata if isinstance(metadata, dict) else {}


def sync_armature_morph_controls(arm):
    if arm is None or getattr(arm, "type", None) != 'ARMATURE':
        raise ValueError("Morph controls require a model armature")
    targets = {}
    hash_names = {}
    mesh_objects = sorted(
        (
            obj for obj in bpy.data.objects
            if getattr(obj, "type", None) == 'MESH'
            and getattr(obj, "parent", None) == arm
            and obj.get("engine_bounds_type", "") != "subset_aabb"
        ),
        key=lambda obj: (int(obj.get("engine_subset_index", 0) or 0), obj.name),
    )
    for obj in mesh_objects:
        shape_keys = getattr(obj.data, "shape_keys", None)
        key_blocks = list(getattr(shape_keys, "key_blocks", []) or [])
        if len(key_blocks) <= 1:
            continue
        metadata = _object_morph_metadata(obj)
        for key in key_blocks[1:]:
            if metadata and str(key.name) not in metadata:
                continue
            stored = metadata.get(str(key.name))
            if isinstance(stored, dict) and stored.get("name"):
                name = str(stored["name"])
                name_hash = int(stored.get("hash", string_crc32(name))) & U32_MASK
                source_index = int(stored.get("index", -1))
            else:
                name = str(key.name)
                name_hash = string_crc32(name)
                source_index = -1
            previous = hash_names.get(name_hash)
            if previous is not None and previous != name:
                raise ValueError(f"Morph target hash collision: {previous!r} and {name!r}")
            hash_names[name_hash] = name
            target = targets.setdefault(name, {
                "name": name,
                "hash": name_hash,
                "source_index": source_index,
                "keys": [],
            })
            target["keys"].append(key)

    ordered = sorted(targets.values(), key=lambda item: (
        0 if int(item["source_index"]) >= 0 else 1,
        int(item["source_index"]) if int(item["source_index"]) >= 0 else int(item["hash"]),
        item["name"],
    ))
    previews = getattr(arm, "engine_model_morph_previews", None)
    previous_values = {}
    if previews is not None:
        previous_values = {
            str(item.target_name): float(item.value)
            for item in previews
            if str(item.target_name or "")
        }
        previews.clear()
    controls = []
    for preview_index, target in enumerate(ordered):
        property_name = f"{MORPH_CONTROL_PREFIX}{int(target['hash']):08x}"
        preview_value = previous_values.get(
            target["name"],
            float(arm.get(property_name, 0.0) or 0.0),
        )
        for key in target["keys"]:
            key.slider_min = -1.0
            key.slider_max = 1.0
            try:
                key.driver_remove("value")
            except Exception:
                pass
            key.value = preview_value
        if property_name in arm:
            try:
                del arm[property_name]
            except Exception:
                pass
        if previews is not None:
            preview = previews.add()
            preview.target_name = target["name"]
            preview.mesh_count = len(target["keys"])
            preview.value = preview_value
        controls.append({
            "name": target["name"],
            "hash": int(target["hash"]),
            "source_index": int(target["source_index"]),
            "preview_index": int(preview_index),
            "mesh_count": len(target["keys"]),
        })
    arm["engine_model_morph_controls_json"] = json.dumps(controls, separators=(",", ":"), sort_keys=True)
    return controls


def _checked_range(data, offset, size, label):
    offset = int(offset)
    size = int(size)
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError(f"{label} is outside the model data")
    return offset


def _read_ascii_z(data, offset, limit=MORPH_NAME_BYTES_MAX + 1):
    _checked_range(data, offset, 1, "morph target name")
    end = data.find(b"\x00", offset, min(len(data), offset + int(limit)))
    if end < 0:
        raise ValueError("morph target name is not null terminated")
    raw = data[offset:end]
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("morph target name is not ASCII") from exc


def _extract_lsb_bits(data, byte_offset, bit_offset, bit_count):
    if bit_count <= 0 or bit_count > 32:
        raise ValueError(f"invalid Morph2 bit count {bit_count}")
    absolute_bit = int(byte_offset) * 8 + int(bit_offset)
    first_byte = absolute_bit >> 3
    shift = absolute_bit & 7
    byte_count = (shift + int(bit_count) + 7) >> 3
    _checked_range(data, first_byte, byte_count, "Morph2 delta bits")
    value = int.from_bytes(data[first_byte:first_byte + byte_count], "little") >> shift
    return value & ((1 << int(bit_count)) - 1)


def _decode_batch(data, batch_offset, masks_base, deltas_base, debug_uncompressed):
    _checked_range(data, batch_offset, MORPH_BATCH_SIZE, "Morph2 batch")
    vertex_start, delta_offset, packed, scale, offset = struct.unpack_from("<IIIff", data, batch_offset)
    mask_index = packed & 0xFFFF
    bit_counts = (
        ((packed >> 16) & 0xF) + 1,
        ((packed >> 21) & 0xF) + 1,
        ((packed >> 26) & 0xF) + 1,
    )
    mask_offset = masks_base + mask_index * 8
    _checked_range(data, mask_offset, 8, "Morph2 valid mask")
    valid_mask = struct.unpack_from("<Q", data, mask_offset)[0]
    delta_count = int(valid_mask).bit_count()
    batch_data = deltas_base + int(delta_offset)
    values = []

    if debug_uncompressed:
        _checked_range(data, batch_data, delta_count * 12, "uncompressed Morph2 deltas")
        for delta_index in range(delta_count):
            values.append((
                struct.unpack_from("<f", data, batch_data + delta_index * 4)[0],
                struct.unpack_from("<f", data, batch_data + (delta_count + delta_index) * 4)[0],
                struct.unpack_from("<f", data, batch_data + (delta_count * 2 + delta_index) * 4)[0],
            ))
    else:
        component_starts = (
            0,
            _align_morph(delta_count * bit_counts[0], 8),
            _align_morph(delta_count * bit_counts[0], 8) + _align_morph(delta_count * bit_counts[1], 8),
        )
        for delta_index in range(delta_count):
            decoded = []
            for axis in range(3):
                quantized = _extract_lsb_bits(
                    data,
                    batch_data,
                    component_starts[axis] + delta_index * bit_counts[axis],
                    bit_counts[axis],
                )
                decoded.append(float(quantized) * float(scale) + float(offset))
            values.append(tuple(decoded))

    result = {}
    value_index = 0
    for local_index in range(MORPH_BATCH_DELTA_COUNT):
        if valid_mask & (1 << local_index):
            result[int(vertex_start) + local_index] = values[value_index]
            value_index += 1
    return result


def decode_model_morph2(data, blocks):
    info_block = blocks.get(BLOCK_HASHES["ModelAnimMorph2Info"])
    if not info_block:
        return None
    geom_block = blocks.get(BLOCK_HASHES["ModelSubsetGeomData"])
    if not geom_block:
        raise ValueError("Morph2 model is missing ModelSubsetGeomData")

    info_base, info_size = info_block
    geom_base, geom_size = geom_block
    if info_size < MORPH_INFO_SIZE:
        raise ValueError(f"Morph2 info is truncated ({info_size} bytes)")
    _checked_range(data, info_base, info_size, "Morph2 info")

    flags, target_count, mirror_count, stitched_count = struct.unpack_from("<4H", data, info_base)
    if stitched_count:
        raise ValueError("stitched Morph2 models are not supported by Blender shape-key import")
    if target_count > MORPH_TARGET_MAX:
        raise ValueError(f"Morph2 target count {target_count} exceeds {MORPH_TARGET_MAX}")
    batch_count, _batch_global, batch_size, _mask_global, mask_size, delta_count, _delta_global, delta_size = struct.unpack_from(
        "<8I", data, info_base + 8
    )
    pointers = struct.unpack_from("<8Q", data, info_base + 40)
    lookup_rel, mirror_rel, targets_rel, subsets_rel, batches_rel, masks_rel, deltas_rel, _stitch_rel = pointers

    def info_ptr(relative, size, label):
        if relative == U64_MASK:
            return None
        if relative + size > info_size:
            raise ValueError(f"{label} exceeds Morph2 info block")
        return info_base + int(relative)

    lookup_base = info_ptr(lookup_rel, (target_count + 1) * 8, "Morph2 target lookup")
    targets_base = info_ptr(targets_rel, target_count * MORPH_TARGET_SIZE, "Morph2 targets")
    if lookup_base is None or targets_base is None:
        raise ValueError("Morph2 target tables are missing")

    lookup = [struct.unpack_from("<II", data, lookup_base + index * 8) for index in range(target_count + 1)]
    if lookup[-1] != (U32_MASK, U32_MASK):
        raise ValueError("Morph2 target lookup sentinel is missing")

    mirror_lookup = []
    if mirror_count:
        mirror_base = info_ptr(mirror_rel, (mirror_count + 1) * 8, "Morph2 mirror lookup")
        if mirror_base is None:
            raise ValueError("Morph2 mirror lookup is missing")
        mirror_lookup = [struct.unpack_from("<II", data, mirror_base + index * 8) for index in range(mirror_count)]
        if struct.unpack_from("<II", data, mirror_base + mirror_count * 8) != (U32_MASK, U32_MASK):
            raise ValueError("Morph2 mirror lookup sentinel is missing")

    raw_targets = []
    subset_record_count = 0
    for target_index in range(target_count):
        target_offset = targets_base + target_index * MORPH_TARGET_SIZE
        name_hash, subset_count, subset_start, extent, name_rel = struct.unpack_from("<IHHf4xQ", data, target_offset)
        if subset_count > MORPH_TARGET_SUBSET_MAX:
            raise ValueError(f"Morph2 target {target_index} has too many subsets")
        name = _read_ascii_z(data, info_base + int(name_rel))
        subset_record_count = max(subset_record_count, int(subset_start) + int(subset_count))
        raw_targets.append({
            "index": target_index,
            "name": name,
            "hash": int(name_hash),
            "subset_start": int(subset_start),
            "subset_count": int(subset_count),
            "extent": float(extent),
        })

    subsets_base = info_ptr(subsets_rel, subset_record_count * MORPH_SUBSET_SIZE, "Morph2 subsets")
    if subset_record_count and subsets_base is None:
        raise ValueError("Morph2 subset table is missing")
    subset_records = [
        struct.unpack_from("<HHI", data, subsets_base + index * MORPH_SUBSET_SIZE)
        for index in range(subset_record_count)
    ]

    if int(batches_rel) + int(batch_size) > geom_size:
        raise ValueError("Morph2 batch data exceeds geometry block")
    if int(masks_rel) + int(mask_size) > geom_size:
        raise ValueError("Morph2 valid masks exceed geometry block")
    if int(deltas_rel) + int(delta_size) > geom_size:
        raise ValueError("Morph2 deltas exceed geometry block")
    batches_base = geom_base + int(batches_rel)
    masks_base = geom_base + int(masks_rel)
    deltas_base = geom_base + int(deltas_rel)
    debug_uncompressed = bool(flags & MORPH_FLAG_DEBUG_UNCOMPRESSED)

    targets = []
    for raw in raw_targets:
        decoded_subsets = []
        for subset_record_index in range(raw["subset_start"], raw["subset_start"] + raw["subset_count"]):
            model_subset_index, target_batch_count, batch_start = subset_records[subset_record_index]
            if int(batch_start) + int(target_batch_count) > int(batch_count):
                raise ValueError(f"Morph2 target {raw['name']} references invalid batches")
            deltas = {}
            for batch_index in range(int(batch_start), int(batch_start) + int(target_batch_count)):
                batch_deltas = _decode_batch(
                    data,
                    batches_base + batch_index * MORPH_BATCH_SIZE,
                    masks_base,
                    deltas_base,
                    debug_uncompressed,
                )
                overlap = set(deltas).intersection(batch_deltas)
                if overlap:
                    raise ValueError(f"Morph2 target {raw['name']} contains overlapping batches")
                deltas.update(batch_deltas)
            decoded_subsets.append({"subset_index": int(model_subset_index), "deltas": deltas})
        target = dict(raw)
        target["subsets"] = decoded_subsets
        targets.append(target)

    return {
        "flags": int(flags),
        "target_count": int(target_count),
        "batch_count": int(batch_count),
        "delta_count": int(delta_count),
        "lookup": lookup[:-1],
        "mirrors": mirror_lookup,
        "targets": targets,
    }


def _morph_name_bytes(name):
    try:
        raw = str(name).encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"Morph target name is not ASCII: {name!r}") from exc
    if not raw or len(raw) > MORPH_NAME_BYTES_MAX:
        raise ValueError(
            f"Morph target name must contain 1-{MORPH_NAME_BYTES_MAX} ASCII bytes: {name!r}"
        )
    return raw


def _append_lsb_bits(buffer, bit_cursor, value, bit_count):
    value = int(value) & ((1 << int(bit_count)) - 1)
    required = (int(bit_cursor) + int(bit_count) + 7) >> 3
    if len(buffer) < required:
        buffer.extend(b"\x00" * (required - len(buffer)))
    remaining = int(bit_count)
    cursor = int(bit_cursor)
    while remaining:
        byte_index = cursor >> 3
        bit_in_byte = cursor & 7
        take = min(remaining, 8 - bit_in_byte)
        mask = (1 << take) - 1
        buffer[byte_index] |= (value & mask) << bit_in_byte
        value >>= take
        cursor += take
        remaining -= take
    return cursor


def _quantization_for_deltas(deltas, precision):
    values = [float(component) for delta in deltas for component in delta]
    minimum = min(values)
    maximum = max(values)
    value_range = maximum - minimum
    if value_range <= 1e-20:
        return 1, 1.0, minimum
    bit_count = 16
    for candidate in range(1, 17):
        scale = value_range / float((1 << candidate) - 1)
        if scale * 0.5 <= float(precision):
            bit_count = candidate
            break
    scale = value_range / float((1 << bit_count) - 1)
    return bit_count, scale, minimum


def _quantize_delta_component(value, bit_count, scale, offset):
    if scale <= 0.0:
        return 0
    quantized = int(math.floor(((float(value) - float(offset)) / float(scale)) + 0.5))
    return max(0, min((1 << int(bit_count)) - 1, quantized))


def _normalize_encode_targets(targets, precision):
    result = []
    hash_names = {}
    for target in targets or []:
        name = str(target.get("name", ""))
        raw_name = _morph_name_bytes(name)
        name_hash = int(target.get("hash", string_crc32(name))) & U32_MASK
        previous = hash_names.get(name_hash)
        if previous is not None and previous != name:
            raise ValueError(f"Morph target hash collision: {previous!r} and {name!r}")
        hash_names[name_hash] = name
        subsets = []
        for subset in target.get("subsets", []):
            filtered = {}
            for vertex_index, delta in subset.get("deltas", {}).items():
                value = (float(delta[0]), float(delta[1]), float(delta[2]))
                if math.sqrt(sum(component * component for component in value)) >= float(precision):
                    filtered[int(vertex_index)] = value
            if filtered:
                subsets.append({"subset_index": int(subset["subset_index"]), "deltas": filtered})
        if subsets:
            if len(subsets) > MORPH_TARGET_SUBSET_MAX:
                raise ValueError(f"Morph target {name!r} exceeds {MORPH_TARGET_SUBSET_MAX} subsets")
            result.append({
                "name": name,
                "name_bytes": raw_name,
                "hash": name_hash,
                "source_index": int(target.get("source_index", -1)),
                "subsets": subsets,
            })
    if len(result) > MORPH_TARGET_MAX:
        raise ValueError(f"Morph target count exceeds {MORPH_TARGET_MAX}")
    source_indices = {}
    for target in result:
        source_index = int(target["source_index"])
        if source_index < 0:
            continue
        previous = source_indices.get(source_index)
        if previous is not None and previous != target["name"]:
            raise ValueError(
                f"Morph targets {previous!r} and {target['name']!r} share source index {source_index}"
            )
        source_indices[source_index] = target["name"]
    result.sort(key=lambda item: (
        0 if int(item["source_index"]) >= 0 else 1,
        int(item["source_index"]) if int(item["source_index"]) >= 0 else item["hash"],
        item["name"],
    ))
    return result


def encode_model_morph2(targets, geom_size, precision=MORPH_DELTA_PRECISION):

    targets = _normalize_encode_targets(targets, precision)
    if not targets:
        return None, b"", {"targets": [], "anim_vertices": {}}

    valid_masks = [U64_MASK]
    valid_mask_lookup = {U64_MASK: 0}
    batches = []
    deltas_blob = bytearray()
    subset_records = []
    delta_count_total = 0
    anim_vertices = {}

    for target in targets:
        target["subset_start"] = len(subset_records)
        target["extent"] = 0.0
        for subset in sorted(target["subsets"], key=lambda item: item["subset_index"]):
            subset_index = int(subset["subset_index"])
            ordered = sorted(subset["deltas"].items())
            target["extent"] = max(
                target["extent"],
                max(max(abs(v) for v in delta) for _index, delta in ordered),
            )
            anim_vertices.setdefault(subset_index, set()).update(index for index, _delta in ordered)
            bit_count, scale, offset = _quantization_for_deltas([delta for _index, delta in ordered], precision)
            subset_batch_start = len(batches)
            subset_delta_start = len(deltas_blob)
            cursor = 0
            while cursor < len(ordered):
                vertex_start = int(ordered[cursor][0])
                group = []
                while cursor < len(ordered) and int(ordered[cursor][0]) - vertex_start < MORPH_BATCH_DELTA_COUNT:
                    group.append(ordered[cursor])
                    cursor += 1
                mask = 0
                for vertex_index, _delta in group:
                    mask |= 1 << (int(vertex_index) - vertex_start)
                mask_index = valid_mask_lookup.get(mask)
                if mask_index is None:
                    mask_index = len(valid_masks)
                    valid_masks.append(mask)
                    valid_mask_lookup[mask] = mask_index
                batch_offset = len(deltas_blob)
                batch_blob = bytearray()
                bit_cursor = 0
                for axis in range(3):
                    for _vertex_index, delta in group:
                        quantized = _quantize_delta_component(delta[axis], bit_count, scale, offset)
                        bit_cursor = _append_lsb_bits(batch_blob, bit_cursor, quantized, bit_count)
                    bit_cursor = _align_morph(bit_cursor, 8)
                    required = bit_cursor >> 3
                    if len(batch_blob) < required:
                        batch_blob.extend(b"\x00" * (required - len(batch_blob)))
                deltas_blob.extend(batch_blob)
                packed = int(mask_index) | ((bit_count - 1) << 16) | ((bit_count - 1) << 21) | ((bit_count - 1) << 26)
                batches.append((vertex_start, batch_offset, packed, float(scale), float(offset)))
                delta_count_total += len(group)
            while len(deltas_blob) & 3:
                deltas_blob.append(0)
            # Batch offsets above are global already because deltas_blob is global.
            subset_records.append((subset_index, len(batches) - subset_batch_start, subset_batch_start))
            if len(deltas_blob) < subset_delta_start:
                raise AssertionError("Morph2 delta packing cursor moved backwards")
        target["subset_count"] = len(subset_records) - target["subset_start"]

    batch_bytes = bytearray()
    for vertex_start, delta_offset, packed, scale, offset in batches:
        batch_bytes.extend(struct.pack("<IIIff", vertex_start, delta_offset, packed, scale, offset))
    while len(batch_bytes) & 31:
        batch_bytes.append(0)
    mask_bytes = bytearray(struct.pack(f"<{len(valid_masks)}Q", *valid_masks))
    while len(mask_bytes) & 15:
        mask_bytes.append(0)
    while len(deltas_blob) & 15:
        deltas_blob.append(0)

    suffix = bytearray()
    batch_geom_offset = _align_morph(geom_size, 32)
    suffix.extend(b"\x00" * (batch_geom_offset - int(geom_size)))
    suffix.extend(batch_bytes)
    mask_geom_offset = batch_geom_offset + len(batch_bytes)
    if mask_geom_offset & 15:
        pad = _align_morph(mask_geom_offset, 16) - mask_geom_offset
        suffix.extend(b"\x00" * pad)
        mask_geom_offset += pad
    suffix.extend(mask_bytes)
    delta_geom_offset = mask_geom_offset + len(mask_bytes)
    if delta_geom_offset & 15:
        pad = _align_morph(delta_geom_offset, 16) - delta_geom_offset
        suffix.extend(b"\x00" * pad)
        delta_geom_offset += pad
    suffix.extend(deltas_blob)

    target_count = len(targets)
    mirror_pairs = []
    hashes = {target["hash"] for target in targets}
    for target in targets:
        name = target["name"]
        if name.startswith("LF_") or name.startswith("RT_"):
            mirror_name = ("RT_" if name.startswith("LF_") else "LF_") + name[3:]
            mirror_hash = string_crc32(mirror_name)
            if mirror_hash in hashes:
                mirror_pairs.append((target["hash"], mirror_hash))
    mirror_pairs.sort()

    cursor = MORPH_INFO_SIZE
    lookup_offset = cursor
    cursor += _align_morph((target_count + 1) * 8, 16)
    mirror_offset = None
    if mirror_pairs:
        mirror_offset = cursor
        cursor += _align_morph((len(mirror_pairs) + 1) * 8, 16)
    targets_offset = cursor
    cursor += _align_morph(target_count * MORPH_TARGET_SIZE, 16)
    subsets_offset = cursor
    cursor += _align_morph(len(subset_records) * MORPH_SUBSET_SIZE, 16)
    cursor += _align_morph(len(mask_bytes), 16)  # builder reserves this area in the info pack
    names_offset = cursor
    name_offsets = []
    for target in targets:
        name_offsets.append(cursor)
        cursor += len(target["name_bytes"]) + 1
    info_size = _align_morph(cursor, 16)
    info = bytearray(info_size)

    struct.pack_into(
        "<4H8I",
        info,
        0,
        0,
        target_count,
        len(mirror_pairs),
        0,
        len(batches),
        0,
        len(batch_bytes),
        0,
        len(mask_bytes),
        delta_count_total,
        0,
        len(deltas_blob),
    )
    pointers = (
        lookup_offset,
        mirror_offset if mirror_offset is not None else U64_MASK,
        targets_offset,
        subsets_offset,
        batch_geom_offset,
        mask_geom_offset,
        delta_geom_offset,
        0,
    )
    struct.pack_into("<8Q", info, 40, *pointers)

    lookup = sorted((target["hash"], index) for index, target in enumerate(targets))
    for index, pair in enumerate(lookup + [(U32_MASK, U32_MASK)]):
        struct.pack_into("<II", info, lookup_offset + index * 8, *pair)
    if mirror_offset is not None:
        for index, pair in enumerate(mirror_pairs + [(U32_MASK, U32_MASK)]):
            struct.pack_into("<II", info, mirror_offset + index * 8, *pair)
    for index, target in enumerate(targets):
        struct.pack_into(
            "<IHHf4xQ",
            info,
            targets_offset + index * MORPH_TARGET_SIZE,
            target["hash"],
            target["subset_count"],
            target["subset_start"],
            target["extent"],
            name_offsets[index],
        )
        start = name_offsets[index]
        info[start:start + len(target["name_bytes"])] = target["name_bytes"]
    for index, record in enumerate(subset_records):
        struct.pack_into("<HHI", info, subsets_offset + index * MORPH_SUBSET_SIZE, *record)

    return bytes(info), bytes(suffix), {
        "targets": targets,
        "anim_vertices": anim_vertices,
        "batch_offset": batch_geom_offset,
        "mask_offset": mask_geom_offset,
        "delta_offset": delta_geom_offset,
        "batch_count": len(batches),
        "delta_count": delta_count_total,
    }


def _smooth_stitches_for_vertices(vertices, position_tolerance, normal_dot_tolerance):
    count = len(vertices)
    if count < 2:
        return []
    tolerance = float(position_tolerance)
    inverse = 1.0 / tolerance
    buckets = {}
    for index, vertex in enumerate(vertices):
        position = vertex["co"]
        cell = tuple(int(math.floor(float(position[axis]) * inverse)) for axis in range(3))
        buckets.setdefault(cell, []).append(index)

    parent = list(range(count))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    neighbor_offsets = tuple(
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
    )
    tolerance_sq = tolerance * tolerance
    for index, vertex in enumerate(vertices):
        position = vertex["co"]
        normal = vertex["normal"]
        cell = tuple(int(math.floor(float(position[axis]) * inverse)) for axis in range(3))
        for dx, dy, dz in neighbor_offsets:
            for other in buckets.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()):
                if other <= index:
                    continue
                other_vertex = vertices[other]
                delta = tuple(float(position[axis]) - float(other_vertex["co"][axis]) for axis in range(3))
                if sum(component * component for component in delta) > tolerance_sq:
                    continue
                other_normal = other_vertex["normal"]
                dot = sum(float(normal[axis]) * float(other_normal[axis]) for axis in range(3))
                if dot >= float(normal_dot_tolerance):
                    union(index, other)

    groups = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    stitches = [indices for indices in groups.values() if len(indices) > 1]
    stitches.sort(key=lambda indices: indices[0])
    for indices in stitches:
        if len(indices) > MORPH_STITCH_INDEX_MAX:
            raise ValueError(
                f"normal-smoothing stitch contains {len(indices)} vertices; Luna supports at most "
                f"{MORPH_STITCH_INDEX_MAX}"
            )
    return stitches


def encode_model_smooth2(
    subsets,
    position_tolerance=SMOOTH_POSITION_TOLERANCE,
    normal_dot_tolerance=SMOOTH_NORMAL_DOT_TOLERANCE,
):
   
    subset_infos = []
    stitches = []
    stitch_indices = []
    for subset in subsets:
        subset_stitches = []
        if int(subset.get("anim_vert_count", 0)) > 0:
            subset_stitches = _smooth_stitches_for_vertices(
                subset.get("vertices", []),
                position_tolerance,
                normal_dot_tolerance,
            )
        stitch_start = len(stitches)
        index_start = len(stitch_indices)
        for indices in subset_stitches:
            local_start = len(stitch_indices) - index_start
            stitches.append((int(local_start) << 16) | len(indices))
            stitch_indices.extend(int(index) for index in indices)
        subset_infos.append((
            len(subset.get("vertices", [])) if subset_stitches else 0,
            len(subset_stitches),
            stitch_start,
            index_start,
        ))

    subset_offset = SMOOTH_INFO_SIZE
    stitches_offset = subset_offset + _align_morph(len(subset_infos) * SMOOTH_SUBSET_SIZE, 16)
    indices_offset = stitches_offset + _align_morph(len(stitches) * 4, 16)
    total_size = indices_offset + _align_morph(len(stitch_indices) * 2, 16)
    output = bytearray(total_size)
    struct.pack_into(
        "<BBHIII",
        output,
        0,
        0,
        0,
        len(subset_infos),
        len(stitches),
        len(stitch_indices),
        0,
    )

    struct.pack_into("<3Q", output, 32, subset_offset, stitches_offset, indices_offset)
    for index, info in enumerate(subset_infos):
        struct.pack_into("<HHII", output, subset_offset + index * SMOOTH_SUBSET_SIZE, *info)
    for index, packed in enumerate(stitches):
        struct.pack_into("<I", output, stitches_offset + index * 4, packed)
    for index, vertex_index in enumerate(stitch_indices):
        if not 0 <= vertex_index <= 0xFFFF:
            raise ValueError(f"normal-smoothing vertex index {vertex_index} exceeds uint16")
        struct.pack_into("<H", output, indices_offset + index * 2, vertex_index)
    return bytes(output), {
        "subset_count": len(subset_infos),
        "stitch_count": len(stitches),
        "stitch_index_count": len(stitch_indices),
    }
