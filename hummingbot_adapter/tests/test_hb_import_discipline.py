"""T7: the runtime modules import only the standard library, numpy, pandas and cvar_mm.gym.formula_as."""

from __future__ import annotations

import ast
import os
import subprocess
import sys

import pytest

from hummingbot_adapter.tests.support import REPO_ROOT

RUNTIME_MODULES = ("features.py", "quotes.py", "policy.py")
THIRD_PARTY = {"numpy", "pandas"}
PROJECT = {"cvar_mm.gym.formula_as"}


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found, problems = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.append(f"relative import at line {node.lineno}")
            found.append(node.module or "")
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if name in {"__import__", "import_module", "exec", "eval"}:
                problems.append(f"dynamic import or evaluation at line {node.lineno}: {name}")
    return found, problems


def _allowed(module: str) -> bool:
    root = module.split(".")[0]
    return module == "__future__" or root in sys.stdlib_module_names or root in THIRD_PARTY or module in PROJECT


@pytest.mark.parametrize("name", RUNTIME_MODULES)
def test_runtime_module_imports_only_the_allowed_modules(name):
    found, problems = _imports(REPO_ROOT / "hummingbot_adapter" / name)
    assert found, "the AST walk found no import"
    assert problems == []
    assert [module for module in found if not _allowed(module)] == []


def test_the_check_rejects_what_the_rule_forbids():
    for module in ("torch", "stable_baselines3.common", "gymnasium", "scipy.stats", "cvar_mm.gym",
                   "cvar_mm.gym.notebook_support", "cvar_mm", "hummingbot_adapter.calibrate"):
        assert not _allowed(module), module
    for module in ("math", "collections.abc", "decimal", "numpy", "pandas", "cvar_mm.gym.formula_as", "__future__"):
        assert _allowed(module), module


def test_runtime_modules_import_without_torch_sb3_gymnasium_or_scipy():
    """As in a container without them: None in sys.modules makes an import fail and find_spec return None,
    so cvar_mm.gym's package init also skips its gymnasium-only imports."""

    code = (
        "import sys\n"
        "for name in ('torch', 'stable_baselines3', 'gymnasium', 'scipy'):\n"
        "    sys.modules[name] = None\n"
        "import hummingbot_adapter.features, hummingbot_adapter.quotes, hummingbot_adapter.policy\n"
        "heavy = {'torch', 'stable_baselines3', 'gymnasium', 'scipy', 'matplotlib', 'seaborn', 'sklearn', 'shap'}\n"
        "loaded = sorted(n for n, m in sys.modules.items() if m is not None and n.split('.')[0] in heavy)\n"
        "cvar_mm = sorted(n for n in sys.modules if n.split('.')[0] == 'cvar_mm')\n"
        "print('ok', loaded, cvar_mm)\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    expected_cvar_mm = ["cvar_mm", "cvar_mm.gym", "cvar_mm.gym.experiment_config", "cvar_mm.gym.formula_as", "cvar_mm.gym.index_names"]
    assert result.stdout.strip() == f"ok [] {expected_cvar_mm}"
