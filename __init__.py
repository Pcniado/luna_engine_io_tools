bl_info = {
    "name": "Luna Engine IO Tools",
    "author": "Pcniado",
    "version": (2, 0, 1),
    "blender": (5, 0, 0),
    "location": "File > Import/Export > Luna Engine Model / Luna Engine Anim",
    "description": "Import/export Luna Engine model files and import/export AnimClip animation data",
    "category": "Import-Export",
}

import importlib
import sys

_MODULE_NAMES = (
    "utils",
    "hashes",
    "constants",
    "binary",
    "dat1",
    "flags",
    "motion",
    "camera_anim",
    "schemas",
    "events",
    "model_import",
    "model_export",
    "anim_import",
    "anim_export",
    "ui",
    "operators",
    "panels",
    "properties",
    "registration",
)


def _load_modules():
    loaded = {}
    for name in _MODULE_NAMES:
        full_name = f"{__name__}.{name}"
        if full_name in sys.modules:
            loaded[name] = importlib.reload(sys.modules[full_name])
        else:
            loaded[name] = importlib.import_module(f".{name}", __name__)
    loaded["registration"]._wire_module_globals()
    return loaded["registration"]


_registration = _load_modules()


def register():
    _registration.register()


def unregister():
    _registration.unregister()


if __name__ == "__main__":
    register()
