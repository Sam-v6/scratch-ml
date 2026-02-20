from pathlib import Path

# This file lives at src/path.py.
# .parent      → src/
# .parent.parent → repo root (SCRATCH_HOME)
SCRATCH_HOME = Path(__file__).resolve().parent.parent
