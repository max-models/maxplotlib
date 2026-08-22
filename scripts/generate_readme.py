#!/usr/bin/env python3
"""Generate README.md from README.qmd using Quarto."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def main() -> int:
    repository_root = Path(__file__).resolve().parent.parent
    quarto = shutil.which("quarto")
    if quarto is None:
        raise SystemExit(
            "Quarto is required to generate README.md. "
            "Install it from https://quarto.org/docs/get-started/"
        )

    subprocess.run(
        [quarto, "render", "README.qmd", "--output", "README.md"],
        cwd=repository_root,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
