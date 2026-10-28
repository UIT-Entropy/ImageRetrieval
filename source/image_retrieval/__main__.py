"""Module entry point for the image retrieval backend.

Example:
    python -m source.image_retrieval --profile local
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from serve import main  # noqa: E402


if __name__ == "__main__":
    main()
