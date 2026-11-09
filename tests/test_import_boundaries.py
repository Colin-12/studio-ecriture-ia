"""Import every source module in isolation and enforce memory's dependency boundary."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"


def source_modules() -> list[str]:
    modules = set()
    for path in SOURCE.rglob("*.py"):
        parts = list(path.relative_to(ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        name = ".".join(parts)
        if name != "src.app.streamlit_app":
            modules.add(name)
    return sorted(modules)


@pytest.mark.parametrize("module", source_modules())
def test_module_imports_in_fresh_isolated_interpreter(module):
    # -I ignores PYTHONPATH and cwd. Preserve the environment, explicitly
    # bootstrap just the repository, and never preload another source module.
    code = f"import sys; sys.path.insert(0, {str(ROOT)!r}); import {module}"
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT,
                            env=os.environ.copy(), capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=90)
    assert result.returncode == 0, f"{module}\n{result.stdout}\n{result.stderr}"


def test_memory_never_imports_agents():
    violations = []
    for path in (SOURCE / "memory").rglob("*.py"):
        package = ".".join(path.relative_to(ROOT).parts[:-1])
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parent = package.split(".")[:len(package.split(".")) - node.level + 1]
                    base = ".".join([*parent, *([base] if base else [])])
                names = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            if any(name == "src.agents" or name.startswith("src.agents.") for name in names):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, "memory imports agents: " + ", ".join(violations)
