from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "backend")
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, check=False)
    return result.returncode


def main() -> int:
    checks = [
        [sys.executable, "-m", "unittest", "discover", "-s", "backend/tests"],
        [sys.executable, "-m", "compileall", "-q", "backend/chat_software"],
    ]
    for command in checks:
        code = run(command)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
