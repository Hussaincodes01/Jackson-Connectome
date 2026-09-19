"""The core must stay pure. If this fails, the two drivers can drift apart
and validation results stop transferring to what appears on screen."""
import ast
import pathlib

FORBIDDEN = ("flybrain.encode", "flybrain.readout", "flybrain.drivers", "flybrain.bridge")
CORE = pathlib.Path("flybrain/core")


def test_core_does_not_import_io_modules():
    offenders = []
    for path in CORE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # For 'from x import y', check the module name
                if node.module and any(node.module.startswith(f) for f in FORBIDDEN):
                    offenders.append(f"{path}: from {node.module}")
                # For relative imports like 'from .. import x', check imported names
                # Relative imports from flybrain package could access forbidden modules
                elif node.module is None and node.level > 0:
                    for alias in node.names:
                        # Check if the imported name itself is a forbidden module,
                        # or if it could be part of a forbidden module path
                        if any(alias.name == f.split('.')[-1] for f in FORBIDDEN):
                            offenders.append(f"{path}: from {'.' * node.level} import {alias.name}")
            elif isinstance(node, ast.Import):
                # For 'import x', check all imported names (not just the first)
                for alias in node.names:
                    if any(alias.name.startswith(f) for f in FORBIDDEN):
                        offenders.append(f"{path}: import {alias.name}")
    assert offenders == [], f"core imports I/O modules: {offenders}"
