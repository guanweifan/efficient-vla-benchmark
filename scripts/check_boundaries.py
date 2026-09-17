"""Check first-party ownership boundaries without importing any runtime."""

import ast
from pathlib import Path


def import_names(path, root):
    parts = path.relative_to(root).with_suffix("").parts
    if "src" not in parts:
        package = ()
    else:
        package = parts[parts.index("src") + 1 : -1]
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = package[: len(package) - node.level + 1] if node.level else ()
            module = (*prefix, *(node.module or "").split("."))
            names.append(".".join(part for part in module if part))
            names.extend(".".join((*module, alias.name)) for alias in node.names)
    return names


def violations(root):
    failures = []
    package = root / "src" / "efficient_vla_benchmark"
    for base in (package, root / "backends"):
        for path in base.rglob("*.py"):
            relative = path.relative_to(root)
            if "upstream" in relative.parts or "tests" in relative.parts:
                continue
            imported = import_names(path, root)
            common = path.is_relative_to(package) and relative.parts[2] != "methods"
            plugin = relative.parts[0] == "backends"
            for name in imported:
                if (
                    "methods" in relative.parts
                    and "implementations" not in relative.parts
                    and name.startswith("efficient_vla_backend_")
                ):
                    failures.append(f"{relative}: shared method code imports a backend")
                if common and (
                    name.startswith(("efficient_vla_backend_", "efficient_vla_benchmark.methods."))
                    or name.split(".")[0] in {"torch", "transformers", "prismatic", "starVLA"}
                ):
                    failures.append(f"{relative}: common code imports runtime-specific {name}")
                if plugin and name.startswith("efficient_vla_benchmark.methods."):
                    failures.append(f"{relative}: backend imports a paper method")
                if "methods" in relative.parts and name.startswith(
                    "efficient_vla_benchmark.methods."
                ):
                    parts = relative.parts
                    own = parts[parts.index("methods") + 2]
                    target = name.split(".")
                    if len(target) > 3 and target[3] != own:
                        failures.append(f"{relative}: imports another method {name}")
    return sorted(set(failures))


if __name__ == "__main__":
    errors = violations(Path(__file__).resolve().parents[1])
    if errors:
        raise SystemExit("\n".join(errors))
    print("First-party import boundaries passed")
