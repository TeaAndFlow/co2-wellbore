from __future__ import annotations

from .cli import build_parser, main
from .orchestrator import run_orchestrator

__all__ = [
    "build_parser",
    "run_orchestrator",
    "main",
]


if __name__ == "__main__":
    main()