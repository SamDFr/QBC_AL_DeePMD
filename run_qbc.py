#!/usr/bin/env python3

"""Validated launcher for the QBC active-learning selector."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import QBC_active_learning_HPC_version as qbc
from validate_environment import main as validate_environment


def main() -> int:
    root = Path(__file__).resolve().parent
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    status = validate_environment()
    if status != 0:
        return status
    qbc.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
