# This module is part of the public-release split of blender_import_model_anim_release.py.
# Behavior-sensitive logic was moved mechanically; avoid algorithm changes here.

from .utils import *

TRIGGER_LOCATOR_JOINT_STRIDE = 4
TRIGGER_PAYLOAD_HEADER_SIZE = 8
TRIGGER_EVENT_OFFSET_SHIFT = 2
TRIGGER_NAME_HASH_OFFSET = 0
TRIGGER_LOC_HASH_OFFSET = 4
TRIGGER_FLAGS_OFFSET = 8
TRIGGER_EVENT_OFFSET_OFFSET = 12
TRIGGER_RADIUS_OFFSET = 14
EVENT_DATA_ALIGN_MASK = 3
NORMALIZED_TRIGGER_TIME_MAX = 65535

def _rebuilt_event_payload_from_stored(action, prefix):
    raw_b64 = str(action.get(f"{prefix}event_payload_b64", "") or "")
    if not raw_b64:
        return None
    try:
        raw_payload = base64.b64decode(raw_b64)
        if len(raw_payload) < DATABUFFER_HEADER_SIZE:
            return None
        _schema_crc, magic, _num_fields, size_object = struct.unpack_from("<IIII", raw_payload, 0)
        if magic != DATABUFFER_MAGIC:
            return None
        payload_size = DATABUFFER_HEADER_SIZE + max(0, int(size_object))
        if payload_size > len(raw_payload):
            return None
        rebuilt, _old_end = rebuild_databuffer_fields(raw_payload[:payload_size], 0, payload_size, action, prefix)
        return rebuilt
    except Exception as exc:
        log_warning("Failed to use copied AnimClip event payload for %s: %s", prefix, exc)
        return None

def rebuild_trigger_data_block(tg_data, tb_data, locator_count, trig_count, action, total_frames=1, old_trig_count=None, frame_start=0):
    entries = _action_trigger_marker_entries(action)
    trig_count = len(entries)

    locator_size = (locator_count + 1) * TRIGGER_LOCATOR_JOINT_STRIDE if locator_count else 0
    locator_data = bytes(tg_data[:locator_size]) if len(tg_data) >= locator_size else (b"\x00" * locator_size)
    event_region = bytearray()
    records = []

    old_records = []
    old_records_start = len(tg_data)
    old_count = old_trig_count if old_trig_count is not None else trig_count
    try:
        _tb_loc_count, _tb_trig_count, old_event_data_size, _tb_marker_count, _tb_data = _read_tracks_counts(tb_data)
        old_records_start = locator_size + int(old_event_data_size)
    except Exception:
        old_records_start = len(tg_data)
    if old_count > 0 and 0 <= old_records_start <= len(tg_data) - (old_count * ANIM_TRIGGER_RECORD_SIZE):
        old_records = [
            bytearray(tg_data[old_records_start + i * ANIM_TRIGGER_RECORD_SIZE:old_records_start + (i + 1) * ANIM_TRIGGER_RECORD_SIZE])
            for i in range(old_count)
        ]
    else:
        old_records_start = len(tg_data)

    total_frames = max(1, int(total_frames))
    for _out_i, (marker_idx, marker) in enumerate(entries):
        prefix = f"marker_{marker_idx}_"
        old_record = old_records[marker_idx] if marker_idx < len(old_records) else bytearray(ANIM_TRIGGER_RECORD_SIZE)
        if len(old_record) < ANIM_TRIGGER_RECORD_SIZE:
            old_record = bytearray(ANIM_TRIGGER_RECORD_SIZE)

        old_name_hash = struct.unpack_from("<I", old_record, TRIGGER_NAME_HASH_OFFSET)[0] if old_record else 0
        old_loc_hash = struct.unpack_from("<I", old_record, TRIGGER_LOC_HASH_OFFSET)[0] if old_record else 0
        old_flags = struct.unpack_from("<H", old_record, TRIGGER_FLAGS_OFFSET)[0] if old_record else 0
        old_rad = struct.unpack_from("<H", old_record, TRIGGER_RADIUS_OFFSET)[0] if old_record else 0
        old_ev_off = (struct.unpack_from("<H", old_record, TRIGGER_EVENT_OFFSET_OFFSET)[0] << TRIGGER_EVENT_OFFSET_SHIFT) if old_record else 0

        old_actor_hash = 0
        old_ev_hash = 0
        old_payload_ok = False
        if old_ev_off + TRIGGER_PAYLOAD_HEADER_SIZE <= old_records_start:
            try:
                old_actor_hash, old_ev_hash = struct.unpack_from("<II", tg_data, old_ev_off)
                old_payload_ok = old_ev_off + TRIGGER_PAYLOAD_HEADER_SIZE < old_records_start
            except struct.error:
                old_payload_ok = False

        name_hash = _idprop_u32(action.get(f"{prefix}name_hash", old_name_hash), old_name_hash)
        loc_hash = _idprop_u32(action.get(f"{prefix}loc_hash", old_loc_hash), old_loc_hash)
        flags = int(action.get(f"{prefix}flags", old_flags)) & 0xFFFF
        rad = int(action.get(f"{prefix}rad", old_rad)) & 0xFFFF
        actor_hash = _idprop_u32(action.get(f"{prefix}actor_hash", old_actor_hash), old_actor_hash)
        ev_hash = _idprop_u32(action.get(f"{prefix}ev_hash", old_ev_hash), old_ev_hash)

        clip_frame = float(marker.frame) - float(frame_start)
        norm_time = int(round((clip_frame / float(total_frames)) * float(NORMALIZED_TRIGGER_TIME_MAX)))
        norm_time = max(0, min(NORMALIZED_TRIGGER_TIME_MAX, norm_time))

        while len(event_region) & EVENT_DATA_ALIGN_MASK:
            event_region.append(0)
        new_ev_off = locator_size + len(event_region)
        event_region.extend(struct.pack("<II", actor_hash, ev_hash))

        force_rebuild = bool(action.get(f"{prefix}rebuild_payload", False))
        stored_payload = None if force_rebuild else _rebuilt_event_payload_from_stored(action, prefix)
        if stored_payload is not None:
            event_region.extend(stored_payload)
        elif force_rebuild or not old_payload_ok:
            event_region.extend(build_databuffer_from_action(action, prefix, ev_hash))
        else:
            rebuilt, _old_end = rebuild_databuffer_fields(
                tg_data, old_ev_off + TRIGGER_PAYLOAD_HEADER_SIZE, old_records_start, action, prefix
            )
            event_region.extend(rebuilt)

        records.append(struct.pack(
            "<IIHHHH",
            name_hash,
            loc_hash,
            flags,
            norm_time,
            (new_ev_off >> TRIGGER_EVENT_OFFSET_SHIFT) & 0xFFFF,
            rad,
        ))

    while len(event_region) & EVENT_DATA_ALIGN_MASK:
        event_region.append(0)

    new_tg_data = locator_data + bytes(event_region) + b"".join(records)
    if len(tb_data) >= 8:
        tb_data = bytearray(tb_data)
        struct.pack_into("<H", tb_data, 2, trig_count)
        struct.pack_into("<I", tb_data, 4, len(event_region))
        tb_data = bytes(tb_data)
    return new_tg_data, tb_data

def _clear_event_ddl_fields(action, prefix):
    old_fields = str(action.get(f"{prefix}ddl_fields", ""))
    for field_path in [f for f in old_fields.split(",") if f]:
        for stem in ("DDL_", "DDLType_", "DDLDeclared_", "DDLEnum_"):
            safe_name = action_ddl_prop_name(prefix, stem, field_path)
            legacy_name = f"{prefix}{stem}{field_path}"
            for key in (safe_name, legacy_name):
                if key in action:
                    del action[key]

def set_action_event_type(action, prefix, event_name, preserve_existing=False):
    event_name = str(event_name or "AnimDamageEvent")
    old_values = {}
    if preserve_existing:
        for field_path in [f for f in str(action.get(f"{prefix}ddl_fields", "")).split(",") if f]:
            prop_name = action_ddl_prop_name(prefix, "DDL_", field_path)
            if prop_name in action:
                old_values[field_path] = action[prop_name]

    _clear_event_ddl_fields(action, prefix)
    action[f"{prefix}event_name"] = event_name
    action[f"{prefix}ev_hash"] = to_signed_32(_event_hash_for_name(event_name))
    action[f"{prefix}rebuild_payload"] = True

    fields = _schema_default_event_fields(event_name)
    action[f"{prefix}ddl_fields"] = ",".join(path for path, _meta, _value in fields)
    for path, meta, value in fields:
        if path in old_values:
            value = old_values[path]
        _store_action_event_field(action, prefix, path, value, meta)

def _action_trigger_marker_entries(action):
    entries = []
    if not action:
        return entries
    pose_markers = getattr(action, "pose_markers", None)
    if not pose_markers:
        return entries
    for marker in pose_markers:
        match = re.match(r"Trigger_\[(\d+)\]_([0-9a-fA-F]+)", marker.name)
        if match:
            entries.append((int(match.group(1)), marker))
    return sorted(entries, key=lambda item: (item[0], item[1].frame, item[1].name))

def _pack_databuffer_scalar_new(ddl_type, value, enum_values=None):
    if ddl_type == DDL_TYPE_BOOLEAN:
        return struct.pack("<?", _parse_bool_value(value))
    if ddl_type == DDL_TYPE_UINT8:
        return struct.pack("<B", _parse_integer_value(value, enum_values) & 0xFF)
    if ddl_type == DDL_TYPE_UINT16:
        return struct.pack("<H", _parse_integer_value(value, enum_values) & 0xFFFF)
    if ddl_type in DDL_U32_STORAGE_TYPES:
        ivalue = _parse_integer_value(value, enum_values)
        if ivalue < 0:
            ivalue = to_unsigned_32(ivalue)
        return struct.pack("<I", ivalue & U32_MASK)
    if ddl_type == DDL_TYPE_UINT64:
        return struct.pack("<Q", _parse_integer_value(value, enum_values) & U64_MASK)
    if ddl_type == DDL_TYPE_INT8:
        return struct.pack("<b", _parse_integer_value(value, enum_values))
    if ddl_type == DDL_TYPE_INT16:
        return struct.pack("<h", _parse_integer_value(value, enum_values))
    if ddl_type == DDL_TYPE_INT32:
        return struct.pack("<i", _parse_integer_value(value, enum_values))
    if ddl_type == DDL_TYPE_INT64:
        return struct.pack("<q", _parse_integer_value(value, enum_values))
    if ddl_type == DDL_TYPE_FLOAT32:
        return struct.pack("<f", float(value))
    if ddl_type == DDL_TYPE_FLOAT64:
        return struct.pack("<d", float(value))
    if ddl_type in DDL_STRING_STORAGE_TYPES:
        encoded = str(value).encode("ascii", errors="replace")
        storage = align4(len(encoded) + 1)
        out = bytearray(DDL_STRING_HEADER_SIZE + storage)
        struct.pack_into("<I", out, 0, len(encoded))
        struct.pack_into("<I", out, 4, string_crc32(encoded.decode("ascii", errors="replace")))
        out[DDL_STRING_HEADER_SIZE:DDL_STRING_HEADER_SIZE + len(encoded)] = encoded
        return bytes(out)
    if ddl_type in DDL_U64_STORAGE_TYPES:
        return struct.pack("<Q", _parse_integer_value(value, enum_values) & U64_MASK)
    return b""

def build_databuffer_from_action(action, prefix, ev_hash_or_name):
    if isinstance(ev_hash_or_name, str):
        event_name = ev_hash_or_name
    else:
        event_name = _event_name_for_hash(_idprop_u32(ev_hash_or_name)) or str(action.get(f"{prefix}event_name", ""))
    schema_crc = _event_hash_for_name(event_name) if event_name else _idprop_u32(ev_hash_or_name, 0)

    field_paths = [f for f in str(action.get(f"{prefix}ddl_fields", "")).split(",") if f]
    encodings = bytearray()
    values = bytearray()
    built_fields = []

    for field_path in field_paths:



        if "." in field_path or "[" in field_path:
            continue
        ddl_type = int(action_ddl_prop_get(action, prefix, "DDLType_", field_path, DDL_TYPE_INT32)) & DDL_TYPE_MASK
        enum_blob = action_ddl_prop_get(action, prefix, "DDLEnum_", field_path, "")
        enum_values = [v for v in str(enum_blob).split("\n") if v] if enum_blob else None
        prop_name = action_ddl_prop_name(prefix, "DDL_", field_path)
        value = action[prop_name] if prop_name in action else _default_value_for_ddl_type(ddl_type)
        name_hash = string_crc32(_clean_field_part(field_path))
        type_count_enc = ((ddl_type & DDL_TYPE_MASK) << DDL_TYPE_SHIFT) | (1 << DDL_COUNT_SHIFT)
        encodings.extend(struct.pack("<II", name_hash, type_count_enc))
        values.extend(_pack_databuffer_scalar_new(ddl_type, value, enum_values))
        built_fields.append(field_path)

    num_fields = len(built_fields)
    fixed_part = bytes(encodings) + (b"\x00" * (num_fields * DATABUFFER_OFFSET_ENTRY_SIZE))
    fixed_part += b"\x00" * (align4(len(fixed_part)) - len(fixed_part))
    while (len(fixed_part) + len(values)) & EVENT_DATA_ALIGN_MASK:
        values.append(0)
    size_object = len(fixed_part) + len(values)
    return struct.pack("<IIII", schema_crc, DATABUFFER_MAGIC, num_fields, size_object) + fixed_part + bytes(values)
