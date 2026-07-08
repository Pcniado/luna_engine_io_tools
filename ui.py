# This module is part of the public-release split of blender_import_model_anim_release.py.
# Behavior-sensitive logic was moved mechanically; avoid algorithm changes here.

from .utils import *

def _draw_event_type_search(layout, operator_idname, owner, marker_name="", selected_label=""):
    row = layout.row(align=True)
    row.prop(owner, "event_search", text="Search")
    q = str(getattr(owner, "event_search", "") or "")
    results = _event_type_search_results(q, limit=10)

    if selected_label:
        small = layout.row(align=True)
        small.scale_y = 0.75
        small.label(text=f"Current: {selected_label}", icon='EVENT_A')

    if not q:
        small = layout.row(align=True)
        small.scale_y = 0.75
        small.label(text="Type to filter. Click a result to apply.", icon='VIEWZOOM')

    if results:
        col = layout.column(align=True)
        col.scale_y = 0.82
        for name in results:
            op = col.operator(operator_idname, text=name, icon='EVENT_A')
            op.event_name = name
            op.event_search = q
            if marker_name:
                op.marker_name = marker_name
    else:
        small = layout.row(align=True)
        small.scale_y = 0.75
        small.label(text="No matches. Type less or check spelling.", icon='ERROR')

    typed = _event_type_best_match(q)
    if typed:
        fields = _schema_default_event_fields(typed)
        small = layout.row(align=True)
        small.scale_y = 0.75
        if fields:
            small.label(text=f"{typed}: {len(fields)} editable field(s)", icon='PROPERTIES')
        else:
            small.label(text=f"{typed}: no schema fields", icon='INFO')

def _active_anim_flag_labels(flags):
    labels = []
    flags = int(flags) & U32_MASK
    for flag in load_flag_registry():
        bit = int(flag.get("bit", 0)) & U32_MASK
        if bit and (flags & bit):
            labels.append(flag.get("label") or _flag_label_from_name(flag.get("name", "")))
    return labels

def _draw_anim_flags_info(layout, scene, arm, action, show_export_toggles=False):
    box = layout.box()
    try:
        box.label(text="Anim Flags")

        original = get_original_flags(action, arm)
        current = get_current_flags(action, arm)

        col = box.column(align=True)
        col.label(text=f"Original: {_hex32(original) if original is not None else 'None'}")
        col.label(text=f"Export: {_hex32(current) if current is not None else 'None'}")

        use_original = bool(getattr(scene, "engine_export_use_original_values", True))
        col.label(text=f"Original File Values: {'On' if use_original else 'Off'}")

        trigger_count = len(_action_trigger_marker_entries(action)) if action else 0
        col.label(text=f"Event Triggers: {trigger_count}")
        col.label(text=f"Has Events Export: {'On' if trigger_count > 0 else 'Off'}")

        flags_for_list = current if current is not None else original
        if flags_for_list is not None:
            labels = _active_anim_flag_labels(flags_for_list)
            list_col = box.column(align=True)
            list_col.label(text="Active Current Flags:")
            if labels:
                for label in labels:
                    list_col.label(text=label)
            else:
                list_col.label(text="None")
        elif not action:
            box.label(text="No imported animation action is active.")

        if show_export_toggles:
            toggle_col = box.column(align=True)
            toggle_col.enabled = not use_original
            toggle_col.prop(scene, "engine_export_looping", text="Looping")
            toggle_col.prop(scene, "engine_export_additive", text="Additive")
            toggle_col.prop(scene, "engine_export_partial", text="Partial")
            toggle_col.prop(scene, "engine_export_partial_motion", text="Partial Motion")
    except Exception as e:
        box.label(text=f"Anim Flags UI error: {e}")

def _draw_root_motion_config(layout, scene, arm):
    box = layout.box()
    box.label(text="Root Motion", icon='ANIM_DATA')
    root_empty = _resolve_root_motion_empty(scene, arm) if arm else None
    source_row = box.row(align=True)
    if hasattr(scene, "engine_root_motion_empty"):
        source_row.prop(scene, "engine_root_motion_empty", text="Motion Empty")
    source_row.operator("anim.create_root_motion_empty", text="Create")
    if root_empty:
        box.label(text=f"Source: {root_empty.name}")
    elif arm:
        box.label(text="No bound root-motion Empty.", icon='ERROR')
    else:
        box.label(text="Select an armature.", icon='INFO')

    box.prop(scene, "engine_root_motion_export_mode", text="Export Mode")
