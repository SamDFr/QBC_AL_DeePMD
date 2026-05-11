#!/usr/bin/env python3

"""Fail-fast validation for the QBC active-learning runtime environment."""

from __future__ import annotations

import glob
import importlib
import os
import sys
from pathlib import Path


REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "ase": "ase",
    "deepmd": "deepmd-kit",
}


def _read_input_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for line in path.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if "#" in raw:
            raw = raw.split("#", 1)[0].strip()
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    problems: list[str] = []
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))

    print("[check] validating Python imports")
    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            problems.append(f"missing import: {module_name} (install package `{package_name}`) -> {exc}")
            continue
        version = getattr(module, "__version__", "unknown")
        print(f"[ok] {module_name} {version}")

    input_path = root / "input.in"
    if not input_path.exists():
        problems.append("missing input.in")
    else:
        print("[check] validating input.in references")
        cfg = _read_input_config(input_path)

        model_value = cfg.get("MODELS", "")
        model_paths = [Path(p.strip()) for p in model_value.split(",") if p.strip()]
        if len(model_paths) < 2:
            problems.append("MODELS must contain at least two model paths")
        else:
            missing_models = [str(path) for path in model_paths if not (root / path).exists()]
            if missing_models:
                problems.append(f"missing model files: {', '.join(missing_models)}")
            else:
                print(f"[ok] found {len(model_paths)} committee model files")
                try:
                    from deepmd.infer import DeepPot

                    DeepPot(str(root / model_paths[0]))
                except Exception as exc:
                    problems.append(
                        "failed to load first DeePMD model with its runtime backend "
                        f"({model_paths[0]}) -> {exc}"
                    )
                else:
                    print(f"[ok] DeePMD backend can load {model_paths[0]}")

        pool_pattern = cfg.get("POOL", "")
        if not pool_pattern:
            problems.append("POOL is not set in input.in")
        else:
            matches = glob.glob(str(root / pool_pattern), recursive=True)
            if not matches:
                problems.append(f"POOL pattern matched no files: {pool_pattern}")
            else:
                print(f"[ok] POOL matched {len(matches)} trajectory file(s)")

        type_map_value = cfg.get("TYPE_MAP", "")
        if not type_map_value:
            problems.append("TYPE_MAP is not set in input.in")
        elif "," not in type_map_value:
            type_map_path = root / type_map_value
            if not type_map_path.exists():
                problems.append(f"TYPE_MAP file does not exist: {type_map_value}")
            else:
                print(f"[ok] TYPE_MAP file exists: {type_map_value}")
        else:
            type_names = [item.strip() for item in type_map_value.split(",") if item.strip()]
            if not type_names:
                problems.append("TYPE_MAP inline list is empty")
            else:
                print(f"[ok] TYPE_MAP inline list has {len(type_names)} entries")

    if problems:
        print("[fail] environment validation found problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("[ok] environment looks ready for QBC_active_learning_HPC_version.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
