import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


def load_script(name: str) -> ModuleType:
    """Import `scripts/<name>.py` as module `name`, because `scripts/` is maintainer tooling, not an importable package.

    The module is registered in `sys.modules` under `name` before its code runs, so a script loaded
    afterwards that runs `import <name>` gets this module.
    """
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
