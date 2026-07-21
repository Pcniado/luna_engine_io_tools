# This module is part of the public-release split of blender_import_model_anim_release.py.
# It centralizes Blender/Python imports and tiny shared helpers used by moved code.

import base64
import hashlib
import json
import logging
import math
import os
import re
import struct
import uuid
from types import SimpleNamespace


def model_topology_signature(indices):
    values = tuple(int(index) & 0xFFFFFFFF for index in indices)
    payload = struct.pack(f"<{len(values)}I", *values) if values else b""
    return hashlib.sha1(payload).hexdigest()


def model_corner_normal_signature(normals):
    digest = hashlib.sha1()
    for normal in normals:
        values = getattr(normal, "vector", normal)
        packed = tuple(
            max(-32767, min(32767, int(round(float(values[axis]) * 32767.0))))
            for axis in range(3)
        )
        digest.update(struct.pack("<3h", *packed))
    return digest.hexdigest()


try:
    import numpy as np
except Exception:  
    np = SimpleNamespace()


def model_shape_key_delta_signature(basis_key, target_key):
    basis_count = len(getattr(basis_key, "data", ()) or ())
    target_count = len(getattr(target_key, "data", ()) or ())
    if basis_count != target_count:
        return ""
    basis = np.empty(basis_count * 3, dtype=np.float32)
    target = np.empty(target_count * 3, dtype=np.float32)
    basis_key.data.foreach_get("co", basis)
    target_key.data.foreach_get("co", target)
    delta = np.asarray(target - basis, dtype="<f4")
    return hashlib.sha1(delta.tobytes(order="C")).hexdigest()

try:
    import mathutils
except Exception:  
    class _DummyVector:
        def __init__(self, values=(0.0, 0.0, 0.0)):
            vals = list(values) if values is not None else [0.0, 0.0, 0.0]
            self.x = vals[0] if len(vals) > 0 else 0.0
            self.y = vals[1] if len(vals) > 1 else 0.0
            self.z = vals[2] if len(vals) > 2 else 0.0
        def copy(self):
            return type(self)((self.x, self.y, self.z))
    class _DummyQuaternion:
        def __init__(self, values=(1.0, 0.0, 0.0, 0.0)):
            vals = list(values) if values is not None else [1.0, 0.0, 0.0, 0.0]
            self.w = vals[0] if len(vals) > 0 else 1.0
            self.x = vals[1] if len(vals) > 1 else 0.0
            self.y = vals[2] if len(vals) > 2 else 0.0
            self.z = vals[3] if len(vals) > 3 else 0.0
        def copy(self):
            return type(self)((self.w, self.x, self.y, self.z))
        def normalized(self):
            return self
        def to_matrix(self):
            return _DummyMatrix()
    class _DummyMatrix:
        @classmethod
        def Rotation(cls, *_args, **_kwargs):
            return cls()
        @classmethod
        def Identity(cls, *_args, **_kwargs):
            return cls()
        @classmethod
        def LocRotScale(cls, *_args, **_kwargs):
            return cls()
        def to_quaternion(self):
            return _DummyQuaternion()
        def to_translation(self):
            return _DummyVector()
        def to_scale(self):
            return _DummyVector((1.0, 1.0, 1.0))
        def to_4x4(self):
            return self
        def inverted(self):
            return self
        def copy(self):
            return type(self)()
        def __matmul__(self, other):
            return other
    mathutils = SimpleNamespace(Vector=_DummyVector, Quaternion=_DummyQuaternion, Matrix=_DummyMatrix)

try:
    import bpy
    from bpy_extras.io_utils import ImportHelper, ExportHelper
    from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, EnumProperty, PointerProperty, CollectionProperty
    from bpy.types import Operator, Panel, PropertyGroup, OperatorFileListElement
except Exception:
    def _property_stub(*_args, **_kwargs):
        return None
    class Operator:
        pass
    class Panel:
        pass
    class PropertyGroup:
        pass
    class ImportHelper:
        pass
    class ExportHelper:
        pass
    class OperatorFileListElement:
        pass
    StringProperty = IntProperty = FloatProperty = BoolProperty = EnumProperty = PointerProperty = CollectionProperty = _property_stub
    class _DummyMenu:
        def append(self, _func):
            pass
        def remove(self, _func):
            pass
    bpy = SimpleNamespace(
        types=SimpleNamespace(Object=type("Object", (), {}), Scene=type("Scene", (), {}), Material=type("Material", (), {}),
                              OperatorFileListElement=OperatorFileListElement,
                              TOPBAR_MT_file_import=_DummyMenu(), TOPBAR_MT_file_export=_DummyMenu()),
        props=SimpleNamespace(StringProperty=StringProperty, IntProperty=IntProperty, FloatProperty=FloatProperty,
                              BoolProperty=BoolProperty, EnumProperty=EnumProperty, PointerProperty=PointerProperty,
                              CollectionProperty=CollectionProperty),
        utils=SimpleNamespace(register_class=lambda _cls: None, unregister_class=lambda _cls: None),
        app=SimpleNamespace(timers=SimpleNamespace(register=lambda *_args, **_kwargs: None)),
        data=SimpleNamespace(objects=[], actions=SimpleNamespace(new=lambda name: None), meshes=SimpleNamespace(remove=lambda *_args, **_kwargs: None), materials=SimpleNamespace(get=lambda _name: None, remove=lambda *_args, **_kwargs: None)),
        context=SimpleNamespace(scene=None, window_manager=SimpleNamespace(popup_menu=lambda *_args, **_kwargs: None)),
    )

_DEBUG_LOG_ENABLED = os.environ.get("LUNA_ENGINE_IO_DEBUG", "").lower() in {"1", "true", "yes", "on"}
_LOGGER = logging.getLogger("luna_engine_io")
if _DEBUG_LOG_ENABLED:
    logging.basicConfig(level=logging.DEBUG)

U32_MASK = 0xFFFFFFFF
U32_SIGN_BIT = 0x80000000
U32_MODULUS = 0x100000000
I32_MAX = 0x7FFFFFFF
U64_MASK = 0xFFFFFFFFFFFFFFFF
U16_MASK = 0xFFFF
I16_SIGN_BIT = 0x8000
I16_MODULUS = 0x10000

def _debug_print(*args, **kwargs):
    if _DEBUG_LOG_ENABLED:
        sep = kwargs.get("sep", " ")
        _LOGGER.debug(sep.join(str(arg) for arg in args))

def log_debug(message, *args):
    if _DEBUG_LOG_ENABLED:
        _LOGGER.debug(message, *args)

def log_warning(message, *args):
    if _DEBUG_LOG_ENABLED:
        _LOGGER.warning(message, *args)

def log_exception(message, *args):
    if _DEBUG_LOG_ENABLED:
        _LOGGER.exception(message, *args)

def _project_root():
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()

def _coerce_hash(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value & U32_MASK
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if re.fullmatch(r"0[xX][0-9a-fA-F]{1,8}", text):
            return int(text, 16) & U32_MASK
        if re.fullmatch(r"[0-9a-fA-F]{8}", text):
            return int(text, 16) & U32_MASK
        try:
            return int(text, 0) & U32_MASK
        except ValueError:
            if re.fullmatch(r"(?:0x)?[0-9a-fA-F]{1,8}", text):
                return int(text.replace("0x", "").replace("0X", ""), 16) & U32_MASK
            return None
    return None

def _humanize_identifier(name):
    if not name:
        return ""
    if name.startswith("field_"):
        return "Unknown 0x" + name[6:]
    suffix = ""
    array_match = re.search(r"(\[\d+\])$", name)
    if array_match:
        suffix = " " + array_match.group(1)
        name = name[:array_match.start()]
    if name.startswith("m_"):
        name = name[2:]
    words = re.sub(r"(?<!^)(?=[A-Z][a-z])", " ", name)
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", words)
    words = words.replace("_", " ").strip()
    return words + suffix

def idprop_path(prop_name):
    return '["' + prop_name.replace("\\", "\\\\").replace('"', '\\"') + '"]'

def to_signed_32(val):
    val = int(val) & U32_MASK
    return val if val < U32_SIGN_BIT else val - U32_MODULUS

def to_unsigned_32(val):
    return int(val) & U32_MASK

def _idprop_u32(value, default=0):
    try:
        ivalue = int(value)
    except Exception:
        return int(default) & U32_MASK
    if ivalue < 0:
        return to_unsigned_32(ivalue)
    return ivalue & U32_MASK

def show_popup(title, message, icon='ERROR'):
    def draw(self, context):
        for line in message.split('\n'):
            self.layout.label(text=line)
    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)

def _sign_extend_16(value):
    value &= U16_MASK
    return value - I16_MODULUS if value & I16_SIGN_BIT else value

def _decode_subset_origin_meters(data, sub_ptr, mpu):
    packed = struct.unpack_from("<iii", data, sub_ptr + 48)
    scale = float(1 << SUBSET_CENTER_LOG_SCALE) * float(mpu)
    return np.array([_sign_extend_16(v) * scale for v in packed], dtype=np.float32)

def _hex32(value):
    # Safe 32-bit hex formatter for signed/unsigned Blender IDProps
    return f"0x{(int(value) & U32_MASK):08X}"

def _resolve_anim_armature(context):
    candidates = []
    for attr in ("object", "active_object"):
        obj = getattr(context, attr, None)
        if obj and obj not in candidates:
            candidates.append(obj)
    for obj in getattr(context, "selected_objects", []) or []:
        if obj and obj not in candidates:
            candidates.append(obj)

    for obj in candidates:
        if getattr(obj, "type", None) == 'ARMATURE':
            return obj
    for obj in candidates:
        parent = getattr(obj, "parent", None)
        if parent and getattr(parent, "type", None) == 'ARMATURE':
            return parent
    for obj in candidates:
        if getattr(obj, "type", None) != 'EMPTY':
            continue
        try:
            binding_id = str(obj.get("engine_root_motion_binding_id", "") or obj.get("engine_clip_binding_id", "") or "")
        except Exception:
            binding_id = ""
        if binding_id:
            matches = []
            for arm in bpy.data.objects:
                if getattr(arm, "type", None) != 'ARMATURE':
                    continue
                try:
                    if str(arm.get("engine_clip_binding_id", "") or "") == binding_id:
                        matches.append(arm)
                except Exception:
                    pass
            if len(matches) == 1:
                return matches[0]
    return None

def _resolve_anim_action(arm):
    if arm and getattr(arm, "animation_data", None):
        return arm.animation_data.action
    return None


def resolve_subset_index_collisions(arm):
   
    seen = {}          # subset_id -> first object that claimed it
    used_ids = set()
    duplicates = []    # objects that need reassignment
    for obj in bpy.data.objects:
        if obj.parent != arm or obj.type != 'MESH' or obj.get("engine_bounds_type", "") == "subset_aabb":
            continue
        try:
            subset_id = int(obj.get("engine_subset_index", -1))
        except Exception:
            subset_id = -1
        if subset_id >= 0:
            if subset_id in seen:
                duplicates.append(obj)
            else:
                seen[subset_id] = obj
            used_ids.add(subset_id)
    if not duplicates:
        return
    next_id = (max(used_ids) + 1) if used_ids else 0
    for obj in duplicates:
        while next_id in used_ids:
            next_id += 1
        obj["engine_subset_index"] = next_id
        obj["engine_lod_mask"] = int(obj.get("engine_lod_mask", 1) or 1)
        used_ids.add(next_id)
        next_id += 1


def model_mesh_subset_ids(arm, resolve_collisions=True):
    if resolve_collisions:
        resolve_subset_index_collisions(arm)
    ids = []
    for obj in bpy.data.objects:
        if obj.parent != arm or obj.type != 'MESH' or obj.get("engine_bounds_type", "") == "subset_aabb":
            continue
        try:
            subset_id = int(obj.get("engine_subset_index", -1))
        except Exception:
            subset_id = -1
        if subset_id >= 0 and subset_id not in ids:
            ids.append(subset_id)
    return sorted(ids)


def _coerce_model_json_list(owner, key):
    try:
        data = json.loads(str(owner.get(key, "[]") or "[]"))
    except Exception:
        data = []
    return data if isinstance(data, list) else []


def _write_model_json_list(owner, key, data):
    owner[key] = json.dumps(data, separators=(",", ":"))


def _unique_ints(values, valid=None):
    result = []
    changed = False
    for value in values or []:
        try:
            number = int(value)
        except Exception:
            changed = True
            continue
        if number < 0 or (valid is not None and number not in valid):
            changed = True
            continue
        if number in result:
            changed = True
            continue
        result.append(number)
    return result, changed


def sanitize_model_look_metadata(arm, mark_modified=False):
    if not arm:
        return False

    valid_subset_ids = set(model_mesh_subset_ids(arm, resolve_collisions=mark_modified))
    looks = _coerce_model_json_list(arm, "engine_model_looks_json")
    groups = _coerce_model_json_list(arm, "engine_model_look_groups_json")
    changed = False

    for look_index, look in enumerate(looks):
        if not isinstance(look, dict):
            looks[look_index] = {
                "index": look_index,
                "name": f"Look {look_index}",
                "subset_ids": [],
                "lods": [{"start": 0, "count": 0} for _ in range(8)],
            }
            changed = True
            continue

        kept_ids, ids_changed = _unique_ints(look.get("subset_ids", []), valid_subset_ids)
        if ids_changed:
            look["subset_ids"] = kept_ids
            look["lods"] = [{"start": 0, "count": len(kept_ids)} for _ in range(8)]
            changed = True
        if int(look.get("index", look_index) or 0) != look_index:
            look["index"] = look_index
            changed = True

    valid_look_indices = set(range(len(looks)))
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            groups[group_index] = {
                "index": group_index,
                "name": f"Look Group {group_index}",
                "look_indices": [],
            }
            changed = True
            continue
        kept_indices, indices_changed = _unique_ints(group.get("look_indices", []), valid_look_indices)
        if indices_changed:
            group["look_indices"] = kept_indices
            changed = True
        if int(group.get("index", group_index) or 0) != group_index:
            group["index"] = group_index
            changed = True

    if changed:
        _write_model_json_list(arm, "engine_model_looks_json", looks)
        _write_model_json_list(arm, "engine_model_look_groups_json", groups)
        if mark_modified:
            arm["engine_model_looks_modified"] = True

    if looks:
        try:
            active_look = int(getattr(arm, "engine_model_active_look", "0"))
        except Exception:
            active_look = 0
        clamped_look = max(0, min(active_look, len(looks) - 1))
        if clamped_look != active_look:
            try:
                arm.engine_model_active_look = str(clamped_look)
            except Exception:
                pass
            changed = True

    if groups:
        try:
            active_group = int(getattr(arm, "engine_model_active_look_group", "0"))
        except Exception:
            active_group = 0
        clamped_group = max(0, min(active_group, len(groups) - 1))
        if clamped_group != active_group:
            try:
                arm.engine_model_active_look_group = str(clamped_group)
            except Exception:
                pass
            changed = True

    return changed
