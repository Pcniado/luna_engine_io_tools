# This module is part of the public-release split of blender_import_model_anim_release.py.
# Behavior-sensitive logic was moved mechanically; avoid algorithm changes here.

from .utils import *

KNOWN_EVENT_HASHES = {}

EVENT_SCHEMAS = {}

FIELD_NAME_TO_META = {}

ENUM_VALUES_BY_TYPE = {}

ENUM_TYPE_HASH_TO_NAME = {}

STRUCT_TYPE_HASH_TO_NAME = {}

SCHEMA_LOAD_STATUS = "DDL schemas not loaded."

DDL_TYPE_UINT8 = 0
DDL_TYPE_UINT16 = 1
DDL_TYPE_UINT32 = 2
DDL_TYPE_UINT64 = 3
DDL_TYPE_INT8 = 4
DDL_TYPE_INT16 = 5
DDL_TYPE_INT32 = 6
DDL_TYPE_INT64 = 7
DDL_TYPE_FLOAT32 = 8
DDL_TYPE_FLOAT64 = 9
DDL_TYPE_STRING = 10
DDL_TYPE_SELECT = 11
DDL_TYPE_BITFIELD = 12
DDL_TYPE_STRUCT = 13
DDL_TYPE_UNKNOWN = 14
DDL_TYPE_BOOLEAN = 15
DDL_TYPE_FILE = 16
DDL_TYPE_TUID = 17
DDL_TYPE_JSON = 18
DDL_TYPE_EMPTY_ARRAY = 19
DDL_TYPE_ASSET_ID = 20

DDL_U32_STORAGE_TYPES = (DDL_TYPE_UINT32, DDL_TYPE_SELECT, DDL_TYPE_BITFIELD)
DDL_STRING_STORAGE_TYPES = (DDL_TYPE_STRING, DDL_TYPE_FILE)
DDL_U64_STORAGE_TYPES = (DDL_TYPE_TUID, DDL_TYPE_ASSET_ID)

DDL_TYPE_MASK = 0x1F
DDL_TYPE_SHIFT = 24
DDL_COUNT_SHIFT = 4
DDL_COUNT_MASK = 0xFFFFF
DDL_FLAGS_MASK = 0xF
DDL_ARRAY_FLAG = 0x1

DDL_STRING_HEADER_SIZE = 16
DATABUFFER_HEADER_SIZE = 16
DATABUFFER_FIELD_ENCODING_SIZE = 8
DATABUFFER_OFFSET_ENTRY_SIZE = 4
DATABUFFER_MAX_FIELD_COUNT = 256
DEFAULT_EVENT_SEARCH_LIMIT = 12
UNSIGNED_DECLARED_TYPES = {"uint32_t", "uint64_t", "uint32", "uint64"}

DDL_TYPE_NAMES = {
    DDL_TYPE_UINT8: "Uint8", DDL_TYPE_UINT16: "Uint16",
    DDL_TYPE_UINT32: "Uint32", DDL_TYPE_UINT64: "Uint64",
    DDL_TYPE_INT8: "Int8", DDL_TYPE_INT16: "Int16",
    DDL_TYPE_INT32: "Int32", DDL_TYPE_INT64: "Int64",
    DDL_TYPE_FLOAT32: "Float32", DDL_TYPE_FLOAT64: "Float64",
    DDL_TYPE_STRING: "String", DDL_TYPE_SELECT: "Select",
    DDL_TYPE_BITFIELD: "Bitfield", DDL_TYPE_STRUCT: "Struct",
    DDL_TYPE_UNKNOWN: "Unknown", DDL_TYPE_BOOLEAN: "Boolean",
    DDL_TYPE_FILE: "File", DDL_TYPE_TUID: "Tuid",
    DDL_TYPE_JSON: "Json", DDL_TYPE_EMPTY_ARRAY: "EmptyArray",
    DDL_TYPE_ASSET_ID: "AssetId"
}

DDL_TYPE_SIZES = {
    DDL_TYPE_UINT8: 1, DDL_TYPE_UINT16: 2,
    DDL_TYPE_UINT32: 4, DDL_TYPE_UINT64: 8,
    DDL_TYPE_INT8: 1, DDL_TYPE_INT16: 2,
    DDL_TYPE_INT32: 4, DDL_TYPE_INT64: 8,
    DDL_TYPE_FLOAT32: 4, DDL_TYPE_FLOAT64: 8,
    DDL_TYPE_SELECT: 4, DDL_TYPE_BITFIELD: 4,
    DDL_TYPE_BOOLEAN: 1,
    DDL_TYPE_TUID: 8, DDL_TYPE_ASSET_ID: 8
}

DATABUFFER_MAGIC = 0x03150044

SYMBOL_CACHE_BASENAME = "blender_import_model_anim_symbols.json"

FIELD_HASH_TO_NAME = {}

BUILTIN_EVENT_NAME_HINTS = (

    "AllowEarlyTransitionEvent",
    "AnimDamageEvent",
    "SyncedAnimConnectEvent",
    "SyncedAnimImpactEvent",
    "SyncedAnimReleaseEvent",
)

def ddl_field_display_label(field_path):
    field_path = resolve_ddl_field_path(field_path)
    leaf = field_path.rsplit(".", 1)[-1]
    leaf_base = re.sub(r"\[\d+\]$", "", leaf)
    meta = FIELD_NAME_TO_META.get(leaf_base, {})
    return meta.get("label") or _humanize_identifier(leaf)

def ddl_field_group_label(field_path):
    field_path = resolve_ddl_field_path(field_path)
    if "." not in field_path:
        return ""
    parts = field_path.rsplit(".", 1)[0].split(".")
    return " / ".join(ddl_field_display_label(part) for part in parts)

IDPROP_NAME_MAX = 63

def action_ddl_prop_name(prefix, stem, field_path):
    full_name = f"{prefix}{stem}{field_path}"
    if len(full_name) <= IDPROP_NAME_MAX:
        return full_name
    return f"{prefix}{stem}{string_crc32(str(field_path)):08X}"

def action_ddl_prop_get(action, prefix, stem, field_path, default=None):
    prop_name = action_ddl_prop_name(prefix, stem, field_path)
    if prop_name in action:
        return action[prop_name]



    legacy_name = f"{prefix}{stem}{field_path}"
    return action.get(legacy_name, default)

def action_ddl_prop_exists(action, prefix, stem, field_path):
    prop_name = action_ddl_prop_name(prefix, stem, field_path)
    if prop_name in action:
        return True
    legacy_name = f"{prefix}{stem}{field_path}"
    return legacy_name in action

def resolve_ddl_field_path(field_path):
    parts = []
    changed = False
    for part in str(field_path).split("."):
        suffix = ""
        array_match = re.search(r"(\[\d+\])$", part)
        if array_match:
            suffix = array_match.group(1)
            base = part[:array_match.start()]
        else:
            base = part

        match = re.fullmatch(r"field_([0-9a-fA-F]{8})", base)
        if match:
            resolved = FIELD_HASH_TO_NAME.get(_coerce_hash(match.group(1)))
            if resolved:
                parts.append(resolved + suffix)
                changed = True
                continue

        parts.append(part)
    return ".".join(parts) if changed else field_path

def migrate_action_field_path(action, prefix, old_path):
    new_path = resolve_ddl_field_path(old_path)
    if new_path == old_path:
        return old_path
    for stem in ("DDL_", "DDLType_", "DDLDeclared_", "DDLEnum_"):
        old_prop = action_ddl_prop_name(prefix, stem, old_path)
        old_legacy_prop = f"{prefix}{stem}{old_path}"
        new_prop = action_ddl_prop_name(prefix, stem, new_path)
        if old_prop in action and new_prop not in action:
            action[new_prop] = action[old_prop]
        elif old_legacy_prop in action and new_prop not in action:
            action[new_prop] = action[old_legacy_prop]
    return new_path

def _clean_field_part(part):
    return re.sub(r"\[\d+\]$", "", part)

def _schema_field(schema_name, field_name):
    schema = EVENT_SCHEMAS.get(schema_name)
    if not isinstance(schema, dict):
        return None
    for field in schema.get("fields", []):
        if field.get("name") == field_name:
            return field
    return None

def _declared_type_for_path(event_name, field_path, values=None):
    parts = [_clean_field_part(p) for p in field_path.split(".") if p]
    if not parts:
        return ""

    schema_name = event_name if event_name in EVENT_SCHEMAS else ""
    last_type = ""
    i = 0
    while i < len(parts):
        part = parts[i]



        if part == "Obj" and values:
            type_path = ".".join(parts[:i] + ["Type"])
            dynamic_type = values.get(type_path)
            if isinstance(dynamic_type, str) and dynamic_type in EVENT_SCHEMAS:
                schema_name = dynamic_type
                i += 1
                continue

        field = _schema_field(schema_name, part) if schema_name else None
        if not field:
            break

        last_type = field.get("type", "")
        if i == len(parts) - 1:
            return last_type
        schema_name = last_type if last_type in EVENT_SCHEMAS else ""
        i += 1

    leaf = parts[-1]
    meta = FIELD_NAME_TO_META.get(leaf, {})
    meta_type = meta.get("type", "")
    if meta_type in {"Select", "Bitfield"} and meta.get("typeHash") is not None:
        type_hash = _coerce_hash(meta.get("typeHash"))
        if type_hash in ENUM_TYPE_HASH_TO_NAME:
            return ENUM_TYPE_HASH_TO_NAME[type_hash]
    return meta_type or last_type

def ddl_enum_values_for_type(type_name):
    if not type_name:
        return []
    return ENUM_VALUES_BY_TYPE.get(type_name, [])

def _extract_meta_string(text, key):
    match = re.search(r"\b" + re.escape(key) + r'\s*\(\s*"((?:[^"\\]|\\.)*)"', text)
    if not match:
        return ""
    try:
        return bytes(match.group(1), "utf-8").decode("unicode_escape")
    except Exception:
        return match.group(1)

def _register_field_name(name, type_name="", meta_text=""):
    if not name:
        return
    FIELD_HASH_TO_NAME.setdefault(string_crc32(name), name)
    meta = FIELD_NAME_TO_META.setdefault(name, {})
    if type_name and "type" not in meta:
        meta["type"] = type_name
    label = _extract_meta_string(meta_text, "label")
    if label:
        meta["label"] = label
    description = _extract_meta_string(meta_text, "description")
    if description:
        meta["description"] = description

def _load_schema_json(schema_path):
    loaded = 0
    with open(schema_path, "r", encoding="utf-8") as f:
        schemas = json.load(f)
    for name, schema in schemas.items():
        event_hash = _coerce_hash(schema.get("hash"))
        if event_hash is not None:
            KNOWN_EVENT_HASHES.setdefault(event_hash, name)
            STRUCT_TYPE_HASH_TO_NAME.setdefault(event_hash, name)
        EVENT_SCHEMAS[name] = schema
        for field in schema.get("fields", []):
            fname = field.get("name")
            if not fname:
                continue
            _register_field_name(fname, field.get("type", ""), "")
            meta = FIELD_NAME_TO_META.setdefault(fname, {})
            if field.get("label"):
                meta["label"] = field["label"]
            if field.get("description"):
                meta["description"] = field["description"]
        loaded += 1
    return loaded

def _load_symbol_cache_json(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    if not isinstance(cache, dict):
        return 0, 0, 0, 0


    if "events" not in cache and "fields" not in cache:
        schema_count = 0
        for name, schema in cache.items():
            if not isinstance(schema, dict):
                continue
            event_hash = _coerce_hash(schema.get("hash"))
            if event_hash is not None:
                KNOWN_EVENT_HASHES.setdefault(event_hash, name)
                STRUCT_TYPE_HASH_TO_NAME.setdefault(event_hash, name)
            EVENT_SCHEMAS[name] = schema
            for field in schema.get("fields", []):
                fname = field.get("name")
                if not fname:
                    continue
                _register_field_name(fname, field.get("type", ""), "")
                meta = FIELD_NAME_TO_META.setdefault(fname, {})
                if field.get("label"):
                    meta["label"] = field["label"]
                if field.get("description"):
                    meta["description"] = field["description"]
            schema_count += 1
        return schema_count, 0, 0, 0

    event_count = 0
    for key, name in cache.get("events", {}).items():
        event_hash = _coerce_hash(key)
        if event_hash is not None and name:
            KNOWN_EVENT_HASHES.setdefault(event_hash, str(name))
            event_count += 1

    field_count = 0
    for key, name in cache.get("fields", {}).items():
        field_hash = _coerce_hash(key)
        if field_hash is not None and name:
            FIELD_HASH_TO_NAME.setdefault(field_hash, str(name))
            FIELD_NAME_TO_META.setdefault(str(name), {})
            field_count += 1

    for name, meta in cache.get("field_meta", {}).items():
        if isinstance(meta, dict):
            FIELD_NAME_TO_META.setdefault(name, {}).update(meta)

    enum_count = 0
    for name, values in cache.get("enums", {}).items():
        if isinstance(values, list):
            ENUM_VALUES_BY_TYPE.setdefault(name, [str(v) for v in values])
            ENUM_TYPE_HASH_TO_NAME.setdefault(string_crc32(name), name)
            enum_count += 1

    schema_count = 0
    for name, schema in cache.get("event_schemas", {}).items():
        if isinstance(schema, dict):
            EVENT_SCHEMAS.setdefault(name, schema)
            schema_hash = _coerce_hash(schema.get("hash"))
            if schema_hash is not None:
                STRUCT_TYPE_HASH_TO_NAME.setdefault(schema_hash, name)
            schema_count += 1

    return event_count, field_count, enum_count, schema_count

def load_ddl_schemas(force=False):
    global SCHEMA_LOAD_STATUS
    if FIELD_HASH_TO_NAME and KNOWN_EVENT_HASHES and not force:
        return

    KNOWN_EVENT_HASHES.clear()
    EVENT_SCHEMAS.clear()
    FIELD_HASH_TO_NAME.clear()
    FIELD_NAME_TO_META.clear()
    ENUM_VALUES_BY_TYPE.clear()
    ENUM_TYPE_HASH_TO_NAME.clear()
    STRUCT_TYPE_HASH_TO_NAME.clear()

    root = _project_root()
    cache_summaries = []
    release_jsons = [
        os.environ.get("ENGINE_SYMBOL_CACHE_JSON", ""),
        os.path.join(root, SYMBOL_CACHE_BASENAME),
        os.environ.get("ENGINE_DDL_SCHEMA_JSON", ""),
    ]

    seen_jsons = set()

    def load_jsons(paths):
        for schema_path in paths:
            if not schema_path or not os.path.exists(schema_path):
                continue
            schema_path = os.path.abspath(schema_path)
            if schema_path in seen_jsons:
                continue
            seen_jsons.add(schema_path)
            try:
                events, fields, enums, schemas = _load_symbol_cache_json(schema_path)
                cache_summaries.append(
                    f"{os.path.basename(schema_path)}: {events} events, {fields} fields, {enums} enums, {schemas} schemas"
                )
            except Exception as e:
                log_warning("Failed to load symbol/schema JSON %s: %s", schema_path, e)

    load_jsons(release_jsons)

    for name in BUILTIN_EVENT_NAME_HINTS:
        KNOWN_EVENT_HASHES.setdefault(string_crc32(name), name)

    SCHEMA_LOAD_STATUS = (
        f"Loaded {len(KNOWN_EVENT_HASHES)} event names and {len(FIELD_HASH_TO_NAME)} field names "
        f"from {len(cache_summaries)} JSON symbol caches."
    )
    if cache_summaries:
        SCHEMA_LOAD_STATUS += " " + "; ".join(cache_summaries[:3])
    SCHEMA_LOAD_STATUS += " DDL source scanning is disabled in the public release."
    log_debug(SCHEMA_LOAD_STATUS)

def _read_databuffer_string(data, pos, data_end):
    if pos + DDL_STRING_HEADER_SIZE > data_end:
        raise struct.error("String header outside DataBuffer")
    length = struct.unpack_from("<I", data, pos)[0]
    text_pos = pos + DDL_STRING_HEADER_SIZE

    storage = align4(length + 1)
    if text_pos + storage > data_end:
        raise struct.error("String payload outside DataBuffer")
    raw = data[text_pos:text_pos + length]
    return raw.decode("ascii", errors="replace"), text_pos + storage

def _read_databuffer_scalar(data, pos, data_end, ddl_type):
    #inspirational
    if ddl_type == DDL_TYPE_BOOLEAN:
        return struct.unpack_from("<?", data, pos)[0], pos + 1
    if ddl_type == DDL_TYPE_UINT8:
        return struct.unpack_from("<B", data, pos)[0], pos + 1
    if ddl_type == DDL_TYPE_UINT16:
        return struct.unpack_from("<H", data, pos)[0], pos + 2
    if ddl_type in DDL_U32_STORAGE_TYPES:
        return struct.unpack_from("<I", data, pos)[0], pos + 4
    if ddl_type == DDL_TYPE_UINT64:
        return struct.unpack_from("<Q", data, pos)[0], pos + 8
    if ddl_type == DDL_TYPE_INT8:
        return struct.unpack_from("<b", data, pos)[0], pos + 1
    if ddl_type == DDL_TYPE_INT16:
        return struct.unpack_from("<h", data, pos)[0], pos + 2
    if ddl_type == DDL_TYPE_INT32:
        return struct.unpack_from("<i", data, pos)[0], pos + 4
    if ddl_type == DDL_TYPE_INT64:
        return struct.unpack_from("<q", data, pos)[0], pos + 8
    if ddl_type == DDL_TYPE_FLOAT32:
        return struct.unpack_from("<f", data, pos)[0], pos + 4
    if ddl_type == DDL_TYPE_FLOAT64:
        return struct.unpack_from("<d", data, pos)[0], pos + 8
    if ddl_type in DDL_STRING_STORAGE_TYPES:
        return _read_databuffer_string(data, pos, data_end)
    if ddl_type in DDL_U64_STORAGE_TYPES:
        return struct.unpack_from("<Q", data, pos)[0], pos + 8
    raise ValueError(f"Unsupported DDL type {ddl_type}")

def _parse_databuffer_object(data, offset, end_offset, path_prefix="", field_meta=None):
    result = {}
    pos = offset
    if pos + DATABUFFER_HEADER_SIZE > end_offset:
        return result, pos

    _schema_crc, magic, num_fields, size_object = struct.unpack_from("<IIII", data, pos)
    pos += DATABUFFER_HEADER_SIZE
    if magic != DATABUFFER_MAGIC or num_fields > DATABUFFER_MAX_FIELD_COUNT:
        return result, offset

    data_end = min(pos + size_object, end_offset)
    if num_fields == 0:
        return result, data_end

    encodings = []
    for _i in range(num_fields):
        if pos + DATABUFFER_FIELD_ENCODING_SIZE > data_end:
            return result, data_end
        name_hash, type_count_enc = struct.unpack_from("<II", data, pos)
        pos += DATABUFFER_FIELD_ENCODING_SIZE
        ddl_type = (type_count_enc >> DDL_TYPE_SHIFT) & DDL_TYPE_MASK
        count = (type_count_enc >> DDL_COUNT_SHIFT) & DDL_COUNT_MASK
        flags = type_count_enc & DDL_FLAGS_MASK
        encodings.append((name_hash, ddl_type, count, bool(flags & DDL_ARRAY_FLAG)))

    pos += num_fields * DATABUFFER_OFFSET_ENTRY_SIZE

    pos = align4(pos)

    for name_hash, ddl_type, count, is_array in encodings:
        field_name = FIELD_HASH_TO_NAME.get(name_hash, f"field_{name_hash:08X}")
        full_name = f"{path_prefix}.{field_name}" if path_prefix else field_name
        element_count = count if count > 0 else 0
        if element_count == 0:
            continue

        try:
            if ddl_type == DDL_TYPE_STRUCT:
                for element_index in range(element_count):
                    element_name = (
                        f"{full_name}[{element_index}]"
                        if is_array or element_count > 1 else full_name
                    )
                    sub_result, pos = _parse_databuffer_object(
                        data, pos, data_end, element_name, field_meta
                    )
                    result.update(sub_result)
                continue

            for element_index in range(element_count):
                value_name = (
                    f"{full_name}[{element_index}]"
                    if is_array or element_count > 1 else full_name
                )
                value, pos = _read_databuffer_scalar(data, pos, data_end, ddl_type)
                result[value_name] = value
                if field_meta is not None:
                    field_meta[value_name] = {
                        "ddl_type": ddl_type,
                        "name_hash": name_hash,
                        "is_array": is_array,
                    }
        except (ValueError, struct.error, IndexError):
            break

        if pos > data_end:
            break

    return result, data_end

def parse_databuffer_fields(data, offset, end_offset, event_name="", include_meta=False):
    #Decode an event DataBuffer payload
    field_meta = {} if include_meta else None
    result, _pos = _parse_databuffer_object(data, offset, end_offset, field_meta=field_meta)
    if include_meta:
        for field_path, meta in field_meta.items():
            declared_type = _declared_type_for_path(event_name, field_path, result)
            enum_values = ddl_enum_values_for_type(declared_type)
            meta["declared_type"] = declared_type
            if enum_values:
                meta["enum_values"] = enum_values
        return result, field_meta
    return result

def _skip_databuffer_scalar(data, pos, data_end, ddl_type):
    _value, new_pos = _read_databuffer_scalar(data, pos, data_end, ddl_type)
    return new_pos

def _patch_databuffer_string(data, pos, data_end, value):
    if pos + DDL_STRING_HEADER_SIZE > data_end:
        return pos
    old_length = struct.unpack_from("<I", data, pos)[0]
    text_pos = pos + DDL_STRING_HEADER_SIZE
    storage = align4(old_length + 1)
    if text_pos + storage > data_end:
        return data_end
    encoded = str(value).encode("ascii", errors="replace")
    capacity = max(0, storage - 1)
    if len(encoded) <= capacity:
        struct.pack_into("<I", data, pos, len(encoded))
        struct.pack_into("<I", data, pos + 4, string_crc32(encoded.decode("ascii", errors="replace")))
        data[text_pos:text_pos + storage] = b"\x00" * storage
        data[text_pos:text_pos + len(encoded)] = encoded
    else:
        log_warning("String value too long for existing AnimClip event storage: %r", value)
    return text_pos + storage

def _parse_integer_value(value, enum_values=None):
    if isinstance(value, str):
        text = value.strip()
        if enum_values and text in enum_values:
            return enum_values.index(text)
        return int(text, 0)
    return int(value)

def _parse_bool_value(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

def _patch_databuffer_scalar(data, pos, data_end, ddl_type, value, enum_values=None):
    if ddl_type == DDL_TYPE_BOOLEAN:
        struct.pack_into("<?", data, pos, _parse_bool_value(value))
        return pos + 1
    if ddl_type == DDL_TYPE_UINT8:
        struct.pack_into("<B", data, pos, _parse_integer_value(value, enum_values) & 0xFF)
        return pos + 1
    if ddl_type == DDL_TYPE_UINT16:
        struct.pack_into("<H", data, pos, _parse_integer_value(value, enum_values) & 0xFFFF)
        return pos + 2
    if ddl_type in DDL_U32_STORAGE_TYPES:
        ivalue = _parse_integer_value(value, enum_values)
        if ivalue < 0:
            ivalue = to_unsigned_32(ivalue)
        struct.pack_into("<I", data, pos, ivalue & U32_MASK)
        return pos + 4
    if ddl_type == DDL_TYPE_UINT64:
        struct.pack_into("<Q", data, pos, _parse_integer_value(value, enum_values) & U64_MASK)
        return pos + 8
    if ddl_type == DDL_TYPE_INT8:
        struct.pack_into("<b", data, pos, _parse_integer_value(value, enum_values))
        return pos + 1
    if ddl_type == DDL_TYPE_INT16:
        struct.pack_into("<h", data, pos, _parse_integer_value(value, enum_values))
        return pos + 2
    if ddl_type == DDL_TYPE_INT32:
        struct.pack_into("<i", data, pos, _parse_integer_value(value, enum_values))
        return pos + 4
    if ddl_type == DDL_TYPE_INT64:
        struct.pack_into("<q", data, pos, _parse_integer_value(value, enum_values))
        return pos + 8
    if ddl_type == DDL_TYPE_FLOAT32:
        struct.pack_into("<f", data, pos, float(value))
        return pos + 4
    if ddl_type == DDL_TYPE_FLOAT64:
        struct.pack_into("<d", data, pos, float(value))
        return pos + 8
    if ddl_type in DDL_STRING_STORAGE_TYPES:
        return _patch_databuffer_string(data, pos, data_end, value)
    if ddl_type in DDL_U64_STORAGE_TYPES:
        struct.pack_into("<Q", data, pos, _parse_integer_value(value, enum_values) & U64_MASK)
        return pos + 8
    return pos

def patch_databuffer_fields(data, offset, end_offset, action, prefix, path_prefix=""):
    pos = offset
    if pos + DATABUFFER_HEADER_SIZE > end_offset:
        return pos

    _schema_crc, magic, num_fields, size_object = struct.unpack_from("<IIII", data, pos)
    pos += DATABUFFER_HEADER_SIZE
    if magic != DATABUFFER_MAGIC or num_fields > DATABUFFER_MAX_FIELD_COUNT:
        return offset

    data_end = min(pos + size_object, end_offset)
    encodings = []
    for _i in range(num_fields):
        if pos + DATABUFFER_FIELD_ENCODING_SIZE > data_end:
            return data_end
        name_hash, type_count_enc = struct.unpack_from("<II", data, pos)
        pos += DATABUFFER_FIELD_ENCODING_SIZE
        ddl_type = (type_count_enc >> DDL_TYPE_SHIFT) & DDL_TYPE_MASK
        count = (type_count_enc >> DDL_COUNT_SHIFT) & DDL_COUNT_MASK
        flags = type_count_enc & DDL_FLAGS_MASK
        encodings.append((name_hash, ddl_type, count, bool(flags & DDL_ARRAY_FLAG)))

    pos += num_fields * DATABUFFER_OFFSET_ENTRY_SIZE
    pos = align4(pos)

    for name_hash, ddl_type, count, is_array in encodings:
        field_name = FIELD_HASH_TO_NAME.get(name_hash, f"field_{name_hash:08X}")
        full_name = f"{path_prefix}.{field_name}" if path_prefix else field_name
        element_count = count if count > 0 else 0
        if element_count == 0:
            continue

        try:
            if ddl_type == DDL_TYPE_STRUCT:
                for element_index in range(element_count):
                    element_name = (
                        f"{full_name}[{element_index}]"
                        if is_array or element_count > 1 else full_name
                    )
                    pos = patch_databuffer_fields(data, pos, data_end, action, prefix, element_name)
                continue

            for element_index in range(element_count):
                value_name = (
                    f"{full_name}[{element_index}]"
                    if is_array or element_count > 1 else full_name
                )
                prop_name = action_ddl_prop_name(prefix, "DDL_", value_name)
                legacy_field_name = f"field_{name_hash:08X}"
                legacy_value_name = (
                    f"{path_prefix}.{legacy_field_name}" if path_prefix else legacy_field_name
                )
                if is_array or element_count > 1:
                    legacy_value_name = f"{legacy_value_name}[{element_index}]"
                legacy_prop_name = action_ddl_prop_name(prefix, "DDL_", legacy_value_name)
                prop_to_use = prop_name if prop_name in action else legacy_prop_name
                if prop_to_use in action:
                    enum_blob = (
                        action_ddl_prop_get(action, prefix, "DDLEnum_", value_name, "")
                        or action_ddl_prop_get(action, prefix, "DDLEnum_", legacy_value_name, "")
                    )
                    enum_values = [v for v in str(enum_blob).split("\n") if v] if enum_blob else None
                    pos = _patch_databuffer_scalar(
                        data, pos, data_end, ddl_type, action[prop_to_use], enum_values
                    )
                else:
                    pos = _skip_databuffer_scalar(data, pos, data_end, ddl_type)
        except (ValueError, struct.error, IndexError):
            break

        if pos > data_end:
            break

    return data_end

def _build_databuffer_string(old_data, pos, data_end, value):
    if pos + DDL_STRING_HEADER_SIZE > data_end:
        return bytes(old_data[pos:data_end]), data_end
    old_length = struct.unpack_from("<I", old_data, pos)[0]
    text_pos = pos + DDL_STRING_HEADER_SIZE
    old_storage = align4(old_length + 1)
    if text_pos + old_storage > data_end:
        return bytes(old_data[pos:data_end]), data_end

    text = str(value)
    encoded = text.encode("ascii", errors="replace")
    storage = align4(len(encoded) + 1)
    out = bytearray(DDL_STRING_HEADER_SIZE + storage)
    struct.pack_into("<I", out, 0, len(encoded))
    struct.pack_into("<I", out, 4, string_crc32(encoded.decode("ascii", errors="replace")))
    out[8:DDL_STRING_HEADER_SIZE] = old_data[pos + 8:pos + DDL_STRING_HEADER_SIZE]
    out[DDL_STRING_HEADER_SIZE:DDL_STRING_HEADER_SIZE + len(encoded)] = encoded
    return bytes(out), text_pos + old_storage

def _build_databuffer_scalar(old_data, pos, data_end, ddl_type, value, enum_values=None):
    if ddl_type == DDL_TYPE_BOOLEAN:
        return struct.pack("<?", _parse_bool_value(value)), pos + 1
    if ddl_type == DDL_TYPE_UINT8:
        return struct.pack("<B", _parse_integer_value(value, enum_values) & 0xFF), pos + 1
    if ddl_type == DDL_TYPE_UINT16:
        return struct.pack("<H", _parse_integer_value(value, enum_values) & 0xFFFF), pos + 2
    if ddl_type in DDL_U32_STORAGE_TYPES:
        ivalue = _parse_integer_value(value, enum_values)
        if ivalue < 0:
            ivalue = to_unsigned_32(ivalue)
        return struct.pack("<I", ivalue & U32_MASK), pos + 4
    if ddl_type == DDL_TYPE_UINT64:
        return struct.pack("<Q", _parse_integer_value(value, enum_values) & U64_MASK), pos + 8
    if ddl_type == DDL_TYPE_INT8:
        return struct.pack("<b", _parse_integer_value(value, enum_values)), pos + 1
    if ddl_type == DDL_TYPE_INT16:
        return struct.pack("<h", _parse_integer_value(value, enum_values)), pos + 2
    if ddl_type == DDL_TYPE_INT32:
        return struct.pack("<i", _parse_integer_value(value, enum_values)), pos + 4
    if ddl_type == DDL_TYPE_INT64:
        return struct.pack("<q", _parse_integer_value(value, enum_values)), pos + 8
    if ddl_type == DDL_TYPE_FLOAT32:
        return struct.pack("<f", float(value)), pos + 4
    if ddl_type == DDL_TYPE_FLOAT64:
        return struct.pack("<d", float(value)), pos + 8
    if ddl_type in DDL_STRING_STORAGE_TYPES:
        return _build_databuffer_string(old_data, pos, data_end, value)
    if ddl_type in DDL_U64_STORAGE_TYPES:
        return struct.pack("<Q", _parse_integer_value(value, enum_values) & U64_MASK), pos + 8
    return b"", pos

def rebuild_databuffer_fields(old_data, offset, end_offset, action, prefix, path_prefix=""):
    pos = offset
    if pos + DATABUFFER_HEADER_SIZE > end_offset:
        return b"", offset

    try:
        schema_crc, magic, num_fields, size_object = struct.unpack_from("<IIII", old_data, pos)
    except struct.error:
        return b"", offset
    pos += DATABUFFER_HEADER_SIZE
    if magic != DATABUFFER_MAGIC or num_fields > DATABUFFER_MAX_FIELD_COUNT:
        return b"", offset

    data_end = min(pos + size_object, end_offset)
    encodings = []
    for _i in range(num_fields):
        if pos + DATABUFFER_FIELD_ENCODING_SIZE > data_end:
            return bytes(old_data[offset:data_end]), data_end
        name_hash, type_count_enc = struct.unpack_from("<II", old_data, pos)
        pos += DATABUFFER_FIELD_ENCODING_SIZE
        ddl_type = (type_count_enc >> DDL_TYPE_SHIFT) & DDL_TYPE_MASK
        count = (type_count_enc >> DDL_COUNT_SHIFT) & DDL_COUNT_MASK
        flags = type_count_enc & DDL_FLAGS_MASK
        encodings.append((name_hash, ddl_type, count, bool(flags & DDL_ARRAY_FLAG)))

    fixed_end = align4(pos + num_fields * DATABUFFER_OFFSET_ENTRY_SIZE)
    fixed_part = bytes(old_data[offset + DATABUFFER_HEADER_SIZE:fixed_end])
    pos = fixed_end
    values = bytearray()

    try:
        for name_hash, ddl_type, count, is_array in encodings:
            field_name = FIELD_HASH_TO_NAME.get(name_hash, f"field_{name_hash:08X}")
            full_name = f"{path_prefix}.{field_name}" if path_prefix else field_name
            element_count = count if count > 0 else 0
            if element_count == 0:
                continue

            if ddl_type == DDL_TYPE_STRUCT:
                for element_index in range(element_count):
                    element_name = (
                        f"{full_name}[{element_index}]"
                        if is_array or element_count > 1 else full_name
                    )
                    rebuilt, pos = rebuild_databuffer_fields(
                        old_data, pos, data_end, action, prefix, element_name
                    )
                    values.extend(rebuilt)
                continue

            for element_index in range(element_count):
                value_name = (
                    f"{full_name}[{element_index}]"
                    if is_array or element_count > 1 else full_name
                )
                prop_name = action_ddl_prop_name(prefix, "DDL_", value_name)
                legacy_field_name = f"field_{name_hash:08X}"
                legacy_value_name = (
                    f"{path_prefix}.{legacy_field_name}" if path_prefix else legacy_field_name
                )
                if is_array or element_count > 1:
                    legacy_value_name = f"{legacy_value_name}[{element_index}]"
                legacy_prop_name = action_ddl_prop_name(prefix, "DDL_", legacy_value_name)
                prop_to_use = prop_name if prop_name in action else legacy_prop_name
                current_value = action[prop_to_use] if prop_to_use in action else None
                enum_blob = (
                    action_ddl_prop_get(action, prefix, "DDLEnum_", value_name, "")
                    or action_ddl_prop_get(action, prefix, "DDLEnum_", legacy_value_name, "")
                )
                enum_values = [v for v in str(enum_blob).split("\n") if v] if enum_blob else None

                if current_value is None:
                    current_value, _ignored = _read_databuffer_scalar(old_data, pos, data_end, ddl_type)

                rebuilt, pos = _build_databuffer_scalar(
                    old_data, pos, data_end, ddl_type, current_value, enum_values
                )
                values.extend(rebuilt)

            if pos > data_end:
                break
    except (ValueError, struct.error, IndexError) as e:
        log_warning("Failed to rebuild event payload field %s: %s", path_prefix or "<root>", e)
        return bytes(old_data[offset:data_end]), data_end

    while (len(fixed_part) + len(values)) & 3:
        values.append(0)
    new_size_object = len(fixed_part) + len(values)
    return (
        struct.pack("<IIII", schema_crc, magic, num_fields, new_size_object)
        + fixed_part
        + bytes(values),
        data_end,
    )

def configure_action_idprop_ui(action, prop_name, field_path):
    try:
        leaf = re.sub(r"\[\d+\]$", "", field_path.rsplit(".", 1)[-1])
        meta = FIELD_NAME_TO_META.get(leaf, {})
        description = meta.get("description") or field_path
        action.id_properties_ui(prop_name).update(description=description)
    except Exception:
        pass

def _action_prop_value_for_field(value, meta):
    ddl_type = meta.get("ddl_type")
    declared_type = meta.get("declared_type", "")

    enum_values = meta.get("enum_values") or []
    if ddl_type == DDL_TYPE_SELECT and isinstance(value, int) and 0 <= value < len(enum_values):
        return str(enum_values[value])

    if ddl_type in (DDL_TYPE_UINT32, DDL_TYPE_UINT64, DDL_TYPE_TUID, DDL_TYPE_ASSET_ID) or declared_type in UNSIGNED_DECLARED_TYPES:
        return str(value)

    if isinstance(value, int) and I32_MAX < value <= U32_MASK:
        return to_signed_32(value)
    return value

def _store_action_event_field(action, prefix, field_path, value, meta):
    prop_name = action_ddl_prop_name(prefix, "DDL_", field_path)
    action[prop_name] = _action_prop_value_for_field(value, meta)

    if meta.get("ddl_type") is not None:
        action[action_ddl_prop_name(prefix, "DDLType_", field_path)] = int(meta.get("ddl_type", 0))
    if meta.get("declared_type"):
        action[action_ddl_prop_name(prefix, "DDLDeclared_", field_path)] = str(meta["declared_type"])
    if meta.get("enum_values"):
        action[action_ddl_prop_name(prefix, "DDLEnum_", field_path)] = "\n".join(str(v) for v in meta["enum_values"])

    configure_action_idprop_ui(action, prop_name, field_path)

def _field_is_unsigned_integer(action, prefix, field_path):
    ddl_type = action_ddl_prop_get(action, prefix, "DDLType_", field_path)
    if ddl_type in {DDL_TYPE_UINT32, DDL_TYPE_UINT64, DDL_TYPE_TUID, DDL_TYPE_ASSET_ID}:
        return True

    declared_type = str(action_ddl_prop_get(action, prefix, "DDLDeclared_", field_path, ""))
    if declared_type in UNSIGNED_DECLARED_TYPES:
        return True

    leaf = _clean_field_part(field_path.rsplit(".", 1)[-1])
    meta_type = str(FIELD_NAME_TO_META.get(leaf, {}).get("type", ""))
    return meta_type in UNSIGNED_DECLARED_TYPES

def normalize_action_event_field_value(action, prefix, field_path):
    prop_name = action_ddl_prop_name(prefix, "DDL_", field_path)
    if prop_name not in action:
        return
    if _field_is_unsigned_integer(action, prefix, field_path) and isinstance(action[prop_name], int):
        value = int(action[prop_name])
        if value < 0:
            value = to_unsigned_32(value)
        action[prop_name] = str(value)

EVENT_TYPE_ENUM_ITEMS_CACHE = []

DDL_DECLARED_TYPE_TO_ID = {
    "uint8": DDL_TYPE_UINT8, "uint8_t": DDL_TYPE_UINT8, "Uint8": DDL_TYPE_UINT8,
    "uint16": DDL_TYPE_UINT16, "uint16_t": DDL_TYPE_UINT16, "Uint16": DDL_TYPE_UINT16,
    "uint32": DDL_TYPE_UINT32, "uint32_t": DDL_TYPE_UINT32, "Uint32": DDL_TYPE_UINT32, "Hash": DDL_TYPE_UINT32,
    "uint64": DDL_TYPE_UINT64, "uint64_t": DDL_TYPE_UINT64, "Uint64": DDL_TYPE_UINT64,
    "int8": DDL_TYPE_INT8, "int8_t": DDL_TYPE_INT8, "Int8": DDL_TYPE_INT8,
    "int16": DDL_TYPE_INT16, "int16_t": DDL_TYPE_INT16, "Int16": DDL_TYPE_INT16,
    "int32": DDL_TYPE_INT32, "int32_t": DDL_TYPE_INT32, "Int32": DDL_TYPE_INT32, "int": DDL_TYPE_INT32,
    "int64": DDL_TYPE_INT64, "int64_t": DDL_TYPE_INT64, "Int64": DDL_TYPE_INT64,
    "float": DDL_TYPE_FLOAT32, "float32": DDL_TYPE_FLOAT32, "Float32": DDL_TYPE_FLOAT32,
    "double": DDL_TYPE_FLOAT64, "float64": DDL_TYPE_FLOAT64, "Float64": DDL_TYPE_FLOAT64,
    "string": DDL_TYPE_STRING, "String": DDL_TYPE_STRING,
    "bool": DDL_TYPE_BOOLEAN, "boolean": DDL_TYPE_BOOLEAN, "Boolean": DDL_TYPE_BOOLEAN,
    "file": DDL_TYPE_FILE, "File": DDL_TYPE_FILE,
    "tuid": DDL_TYPE_TUID, "Tuid": DDL_TYPE_TUID,
    "json": DDL_TYPE_JSON, "Json": DDL_TYPE_JSON,
    "assetid": DDL_TYPE_ASSET_ID, "AssetId": DDL_TYPE_ASSET_ID,
}

def _event_name_for_hash(ev_hash):
    return KNOWN_EVENT_HASHES.get(ev_hash) or KNOWN_EVENT_HASHES.get(_idprop_u32(ev_hash)) or ""

def _event_hash_for_name(event_name):
    event_name = str(event_name or "").strip()
    if not event_name:
        return string_crc32("AnimDamageEvent")
    return string_crc32(event_name)

def _event_type_names():
    names = set()
    for name in EVENT_SCHEMAS.keys():
        if isinstance(name, str) and name:
            names.add(name)
    for name in KNOWN_EVENT_HASHES.values():
        if isinstance(name, str) and name and not name.startswith("Unknown_"):
            names.add(name)
    if not names:
        names.add("AnimDamageEvent")
    return sorted(names, key=lambda n: n.lower())

def event_type_enum_items(self, context):
    global EVENT_TYPE_ENUM_ITEMS_CACHE
    EVENT_TYPE_ENUM_ITEMS_CACHE = [
        (name, name, f"0x{_event_hash_for_name(name):08X}")
        for name in _event_type_names()
    ]
    return EVENT_TYPE_ENUM_ITEMS_CACHE

def _event_type_search_results(query, limit=24):
    names = _event_type_names()
    q = str(query or "").strip().lower()
    if not q:
        preferred = [
            "AnimDamageEvent",
            "AllowEarlyTransitionEvent",
            "SyncedAnimImpactEvent",
            "SyncedAnimConnectEvent",
            "SyncedAnimReleaseEvent",
        ]
        out = []
        name_set = set(names)
        for name in preferred:
            if name in name_set and name not in out:
                out.append(name)
        for name in names:
            if name.endswith("Event") and name not in out:
                out.append(name)
            if len(out) >= min(limit, DEFAULT_EVENT_SEARCH_LIMIT):
                break
        return out

    scored = []
    for name in names:
        low = name.lower()
        if q not in low:
            continue

        if low == q:
            score = 0
        elif low.startswith(q):
            score = 1
        elif re.search(r"(?:^|_|\b)" + re.escape(q), low):
            score = 2
        else:
            score = 3
        has_schema_penalty = 0 if name in EVENT_SCHEMAS else 1
        scored.append((score, has_schema_penalty, len(name), low, name))
    scored.sort()
    return [item[-1] for item in scored[:limit]]

def _event_type_best_match(query):
    q = str(query or "").strip()
    if not q:
        return ""
    names = _event_type_names()
    exact = {n.lower(): n for n in names}
    if q.lower() in exact:
        return exact[q.lower()]
    results = _event_type_search_results(q, limit=2)
    return results[0] if len(results) == 1 else ""

def _ddl_type_for_declared_type(type_name):
    type_name = str(type_name or "")
    if type_name in EVENT_SCHEMAS:
        return DDL_TYPE_STRUCT
    if type_name in ENUM_VALUES_BY_TYPE:
        return DDL_TYPE_SELECT
    if type_name in DDL_DECLARED_TYPE_TO_ID:
        return DDL_DECLARED_TYPE_TO_ID[type_name]
    low = type_name.lower()
    if low in DDL_DECLARED_TYPE_TO_ID:
        return DDL_DECLARED_TYPE_TO_ID[low]
    if "hash" in low or low.endswith("id"):
        return DDL_TYPE_UINT32
    if "float" in low:
        return DDL_TYPE_FLOAT32
    if "bool" in low:
        return DDL_TYPE_BOOLEAN
    if "string" in low or "name" in low:
        return DDL_TYPE_STRING
    return DDL_TYPE_INT32

def _default_value_for_ddl_type(ddl_type, declared_type="", enum_values=None):
    enum_values = enum_values or []
    if ddl_type == DDL_TYPE_SELECT and enum_values:
        return enum_values[0]
    if ddl_type in (DDL_TYPE_FLOAT32, DDL_TYPE_FLOAT64):
        return 0.0
    if ddl_type in (DDL_TYPE_STRING, DDL_TYPE_FILE, DDL_TYPE_JSON):
        return ""
    if ddl_type == DDL_TYPE_BOOLEAN:
        return False
    if ddl_type in (DDL_TYPE_UINT32, DDL_TYPE_UINT64, DDL_TYPE_TUID, DDL_TYPE_ASSET_ID) or declared_type in UNSIGNED_DECLARED_TYPES:
        return "0"
    return 0

def _schema_default_event_fields(event_name, schema_name=None, path_prefix=""):
    schema_name = schema_name or event_name
    schema = EVENT_SCHEMAS.get(schema_name, {})
    out = []
    for field in schema.get("fields", []):
        fname = field.get("name", "")
        if not fname:
            continue
        ftype = field.get("type", "")
        is_array = bool(field.get("is_array"))
        path = f"{path_prefix}.{fname}" if path_prefix else fname
        if is_array:


            continue
        if ftype in EVENT_SCHEMAS:
            out.extend(_schema_default_event_fields(event_name, ftype, path))
        else:
            ddl_type = _ddl_type_for_declared_type(ftype)
            enum_values = ddl_enum_values_for_type(ftype)
            meta = {"ddl_type": ddl_type, "declared_type": ftype}
            if enum_values:
                meta["enum_values"] = enum_values
            out.append((path, meta, _default_value_for_ddl_type(ddl_type, ftype, enum_values)))
    return out
