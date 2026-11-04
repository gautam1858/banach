"""Entry point for the `banach` command: runs the bundled binary with the bundled engine."""

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    bindir = Path(__file__).resolve().parent / "bin"
    exe = bindir / "banach"
    if not exe.exists():
        sys.exit(
            "banach: this wheel has no prebuilt binary for your platform.\n"
            "Build from source: https://github.com/gautam1858/banach#build-from-source"
        )
    env = dict(os.environ)
    engine = bindir / "banach-engine"
    if engine.exists():
        env.setdefault("BANACH_ENGINE", str(engine))
    args = [str(exe)] + sys.argv[1:]
    if os.name == "posix":
        os.execve(str(exe), args, env)
    sys.exit(subprocess.call(args, env=env))
