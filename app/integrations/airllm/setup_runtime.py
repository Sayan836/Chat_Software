"""Create the isolated AirLLM runtime for Chat_Software (user-initiated; downloads ~1-2 GB).

    python setup_runtime.py --home <airllm-home>

Clones the pinned upstream AirLLM into <home>/source, applies the macOS CPU patch, creates
<home>/venv and installs requirements.txt plus AirLLM (editable). Re-running is safe: an existing
checkout with local changes is never overwritten. Model weights are not downloaded here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(args, **kwargs) -> None:
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads((HERE / "upstream.json").read_text())
    home = args.home.resolve()
    home.mkdir(parents=True, exist_ok=True)
    source = home / pin["checkout"]
    if not source.exists():
        run(["git", "clone", pin["repository"], source])
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != pin["commit"]:
        if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
            raise SystemExit("The AirLLM checkout has local changes; inspect it before changing revision.")
        run(["git", "-C", source, "fetch", "origin", pin["commit"]])
        run(["git", "-C", source, "checkout", "--detach", pin["commit"]])
    patch = HERE / "patches" / "cpu-on-macos.patch"
    already = subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)],
                             capture_output=True).returncode == 0
    if not already:
        run(["git", "-C", source, "apply", patch])
    python = home / "venv" / "bin" / "python"
    if not python.exists():
        run([sys.executable, "-m", "venv", home / "venv"])
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "-r", HERE / "requirements.txt", "-e", source / pin["package_directory"]])
    run([python, "-c", "import airllm, torch, transformers; print('airllm runtime ok')"])
    print(f"AirLLM runtime ready: {python}")


if __name__ == "__main__":
    main()
