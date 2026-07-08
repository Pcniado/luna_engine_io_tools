# This module is part of the public-release split of blender_import_model_anim_release.py.
# Behavior-sensitive logic was moved mechanically; avoid algorithm changes here.

from .utils import *
from .constants import *

ANIM_CLIP_FLAG_REGISTRY = []

ANIM_CLIP_FLAG_BY_BIT = {}
ANIM_CLIP_FLAG_WIDTH = 32

_BUILTIN_ANIM_CLIP_FLAGS = [
    ("kFlagsStd", FLAG_STD, "Standard", "Standard skeletal animation clip.", True, False),
    ("kFlagsCurves", FLAG_CURVES, "Curves", "Curve clip type.", False, False),
    ("kFlagsFacialPoses", FLAG_FACIAL_POSES, "Facial Poses", "Facial pose clip type.", False, False),
    ("kFlagsPerformance", FLAG_PERFORMANCE, "Performance", "Performance clip type.", False, False),
    ("kFlagsCamera", FLAG_CAMERA, "Camera", "Camera clip type.", False, False),
    ("kFlagsIsLooping", FLAG_LOOPING, "Looping", "Clip loops at the end.", True, False),
    ("kFlagsIsAdditive", FLAG_IS_ADDITIVE, "Additive", "Additive pose animation.", True, False),
    ("kFlagsIsPartial", FLAG_IS_PARTIAL, "Partial", "Partial-body animation.", True, False),
    ("kFlagsHasPhase", FLAG_HAS_PHASE, "Has Phase", "Clip carries phase data.", False, True),
    ("kFlagsConstantPhase", FLAG_CONSTANT_PHASE, "Constant Phase", "Clip has constant phase data.", False, True),
    ("kFlagsIsEmbedded", FLAG_IS_EMBEDDED, "Embedded", "Clip data is embedded in a parent asset.", False, True),
    ("kFlagsPartialMotion", FLAG_PARTIAL_MOTION, "Partial Motion", "Partial animation explicitly applies motion.", True, False),
    ("kFlagsLocator", FLAG_LOCATOR, "Locator", "Locator clip type.", False, False),
    ("kFlagsUserPose", FLAG_USER_POSE, "User Pose", "Runtime generated pose flag.", False, True),
    ("kFlagsFrameDataLookup", FLAG_FRAME_DATA_LOOKUP, "Frame Lookup", "Uses a frame-data lookup table.", False, True),
    ("kFlagsStreamFrameData", FLAG_STREAM_FRAME_DATA, "Streamed Frames", "Frame data is streamed.", False, True),
    ("kFlagsCinematic", FLAG_CINEMATIC, "Cinematic", "Cinematic animation clip.", False, False),
    ("kFlagsUncompressed", FLAG_UNCOMPRESSED, "Uncompressed", "Uncompressed clip data.", False, False),
    ("kAnimFlagsHasAnimMorph", FLAG_HAS_ANIM_MORPH, "Has Morph", "Clip has morph animation data.", False, True),
    ("kAnimFlagsHasAnimGeom", FLAG_HAS_ANIM_GEOM, "Has Anim Geom", "Clip has animated geometry data.", False, True),
    ("kAnimFlagsHasEvents", FLAG_HAS_EVENTS, "Has Events", "Clip has animation event data.", False, False),
    ("kAnimFlagsHasMotion", FLAG_HAS_MOTION, "Has Motion", "Clip has gameplay/root motion data.", False, True),
    ("kAnimFlagsHasCustomTracks", FLAG_HAS_CUSTOM_TRACKS, "Has Custom Tracks", "Clip has custom tracks.", False, True),
    ("kAnimFlagsHasFacial", FLAG_HAS_FACIAL, "Has Facial", "Clip participates in facial processing.", False, True),
    ("kAnimFlagsHasAnimZiva", FLAG_HAS_ANIM_ZIVA, "Has Anim Ziva", "Clip has simulation animation data.", False, True),
    ("kAnimFlagsHasMotionSamples", FLAG_HAS_MOTION_SAMPLES, "Has Motion Samples", "Runtime/sample motion indicator.", False, True),
    ("kAnimFlagsHasGeomCache", FLAG_HAS_GEOM_CACHE, "Has Geom Cache", "Clip has geometry-cache animation.", False, True),
    ("kAnimFlagMotionDelta", FLAG_MOTION_DELTA, "Motion Delta", "Motion-delta processing flag.", False, True),
]

def _signed_idprop_u32(value):
    value = int(value) & U32_MASK
    return value if value < U32_SIGN_BIT else value - U32_MODULUS

def _read_idprop_u32(container, key, default=None):
    if not container or key not in container:
        return default
    try:
        value = int(container.get(key))
    except Exception:
        return default
    return value & U32_MASK

def _flag_label_from_name(name):
    text = str(name or "")
    for prefix in ("AnimClip::", "AnimClip.", "kAnimFlags", "kFlags", "FLAG_", "Flag"):
        text = text.replace(prefix, "")
    text = re.sub(r"(?<!^)(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = text.replace("_", " ").strip()
    return text or str(name)

def _register_anim_clip_flag(name, bit, label="", description="", editable=False, dangerous=False, source="builtin"):
    bit = _coerce_hash(bit)
    if bit is None or bit == 0:
        return
    existing = ANIM_CLIP_FLAG_BY_BIT.get(bit)
    if existing:
        if not existing.get("name") and name:
            existing["name"] = str(name)
        if not existing.get("label") and label:
            existing["label"] = str(label)
        existing["editable"] = bool(existing.get("editable") or editable)
        existing["dangerous"] = bool(existing.get("dangerous") or dangerous)
        return
    meta = {
        "name": str(name),
        "bit": bit,
        "label": str(label or _flag_label_from_name(name)),
        "description": str(description or ""),
        "source": source,
        "known": True,
        "editable": bool(editable),
        "dangerous": bool(dangerous),
    }
    ANIM_CLIP_FLAG_REGISTRY.append(meta)
    ANIM_CLIP_FLAG_BY_BIT[bit] = meta

def _iter_symbol_flag_pairs(symbols):
    if not isinstance(symbols, dict):
        return
    for key in ("flags", "constants"):
        values = symbols.get(key)
        if isinstance(values, dict):
            for name, bit in values.items():
                if "Anim" in str(name) and ("Flag" in str(name) or "Flags" in str(name)):
                    yield name, bit
    values = symbols.get("AnimClipFlags")
    if isinstance(values, dict):
        for name, bit in values.items():
            yield name, bit
    elif isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                yield item.get("name"), item.get("value", item.get("bit"))
    bitfields = symbols.get("bitfields")
    if isinstance(bitfields, dict):
        values = bitfields.get("AnimClipFlags") or bitfields.get("AnimClip::Flags")
        if isinstance(values, dict):
            for name, bit in values.items():
                yield name, bit
        elif isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    yield item.get("name"), item.get("value", item.get("bit"))

def load_flag_registry(symbols=None):
    if not ANIM_CLIP_FLAG_REGISTRY:
        for name, bit, label, description, editable, dangerous in _BUILTIN_ANIM_CLIP_FLAGS:
            _register_anim_clip_flag(name, bit, label, description, editable, dangerous, "builtin")
    if symbols:
        for name, bit in _iter_symbol_flag_pairs(symbols) or ():
            if name is None or bit is None:
                continue
            dangerous = _coerce_hash(bit) in MOTION_CLIP_FLAG_BITS
            _register_anim_clip_flag(name, bit, source="symbols", dangerous=dangerous)
    return ANIM_CLIP_FLAG_REGISTRY

def known_flag_mask(include_noneditable=True):
    load_flag_registry()
    mask = 0
    for flag in ANIM_CLIP_FLAG_REGISTRY:
        if include_noneditable or flag.get("editable"):
            mask |= int(flag["bit"]) & U32_MASK
    return mask & U32_MASK

def get_unknown_set_bits(flags, registry=None):
    registry = registry or load_flag_registry()
    known = 0
    for flag in registry:
        known |= int(flag["bit"]) & U32_MASK
    unknown = (int(flags) & U32_MASK) & ~known
    return [1 << bit for bit in range(ANIM_CLIP_FLAG_WIDTH) if unknown & (1 << bit)]

def describe_flags(flags):
    registry = load_flag_registry()
    out = []
    flags = int(flags) & U32_MASK
    for flag in sorted(registry, key=lambda item: int(item["bit"])):
        if flags & int(flag["bit"]):
            out.append(flag["label"])
    for bit in get_unknown_set_bits(flags, registry):
        out.append(f"Unknown Bit 0x{bit:08X}")
    return out

def set_original_flags(container, flags):
    if not container:
        return
    flags = int(flags) & U32_MASK
    container["engine_clip_flags_original"] = _signed_idprop_u32(flags)
    container["engine_clip_flags_current"] = _signed_idprop_u32(flags)
    container["engine_clip_flags_source"] = "imported"
    container["engine_clip_flags"] = _signed_idprop_u32(flags)

def migrate_clip_flags(*containers):
    for container in containers:
        if not container:
            continue
        old_flags = _read_idprop_u32(container, "engine_clip_flags")
        if old_flags is None:
            continue
        if "engine_clip_flags_original" not in container:
            container["engine_clip_flags_original"] = _signed_idprop_u32(old_flags)
        if "engine_clip_flags_current" not in container:
            container["engine_clip_flags_current"] = _signed_idprop_u32(old_flags)
        if "engine_clip_flags_source" not in container:
            container["engine_clip_flags_source"] = "migrated"

def get_original_flags(*containers):
    for container in containers:
        migrate_clip_flags(container)
        value = _read_idprop_u32(container, "engine_clip_flags_original")
        if value is not None:
            return value
    return None

def get_current_flags(*containers):
    for container in containers:
        migrate_clip_flags(container)
        value = _read_idprop_u32(container, "engine_clip_flags_current")
        if value is not None:
            return value
    original = get_original_flags(*containers)
    return original if original is not None else FLAG_DEFAULT_EXPORT

def set_current_flags(container, flags):
    if container:
        container["engine_clip_flags_current"] = _signed_idprop_u32(flags)

def apply_flag_toggles(base_flags, toggles, registry=None):
    flags = int(base_flags) & U32_MASK
    bit_by_name = {flag["name"]: int(flag["bit"]) for flag in (registry or load_flag_registry())}
    named_bits = {
        "looping": FLAG_LOOPING,
        "additive": FLAG_IS_ADDITIVE,
        "partial": FLAG_IS_PARTIAL,
        "partial_motion": FLAG_PARTIAL_MOTION,
    }
    for name, bit in named_bits.items():
        if name not in toggles:
            continue
        if toggles[name]:
            flags |= bit_by_name.get(name, bit)
        else:
            flags &= ~bit_by_name.get(name, bit)
    return flags & U32_MASK

def sync_scene_flag_toggles_from_flags(scene, flags):
    if not scene:
        return
    flags = int(flags) & U32_MASK
    if hasattr(scene, "engine_export_looping"):
        scene.engine_export_looping = bool(flags & FLAG_LOOPING)
        scene.engine_export_additive = bool(flags & FLAG_IS_ADDITIVE)
        scene.engine_export_partial = bool(flags & FLAG_IS_PARTIAL)
        scene.engine_export_partial_motion = bool(flags & FLAG_PARTIAL_MOTION)
