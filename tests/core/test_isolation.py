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
            if isinstance(node, ast.ImportFrom) and node.module:
                name = node.module
            elif isinstance(node, ast.Import):
                name = node.names[0].name
            else:
                continue
            if any(name.startswith(f) for f in FORBIDDEN):
                offenders.append(f"{path}: {name}")
    assert offenders == [], f"core imports I/O modules: {offenders}"
