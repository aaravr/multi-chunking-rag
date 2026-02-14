"""WO-010: Assert no pickle usage in codebase (RCE mitigation)."""

import ast
import os


def test_no_pickle_import_in_source():
    """No 'import pickle' or 'from pickle import' in application code."""
    skip_dirs = {"__pycache__", ".git", "venv", ".venv", "node_modules"}
    skip_files = {"test_wo010_no_pickle.py"}
    violations = []
    for root, _, files in os.walk("."):
        root_rel = root.replace("\\", "/")
        if any(f"/{s}/" in f"/{root_rel}/" or root_rel.startswith(s) for s in skip_dirs):
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            if name in skip_files:
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r") as f:
                    tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "pickle":
                                violations.append(path)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module == "pickle":
                            violations.append(path)
            except (SyntaxError, UnicodeDecodeError):
                pass
    assert not violations, f"Pickle import found: {violations}"
