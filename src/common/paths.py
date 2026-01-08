#!/usr/bin/env python
"""
Creates paths that are imported downstream for utility.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SRC_HOME = PROJECT_ROOT / "src"
DATA_HOME = PROJECT_ROOT / "data"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts"
ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
