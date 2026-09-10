"""Run an isolated Hermes one-shot baseline from a prompt file.

This helper intentionally avoids shell quoting: the prompt is read from disk
and Hermes is invoked with an argv list.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-bin", required=True, type=Path)
    parser.add_argument("--hermes-home", required=True, type=Path)
    parser.add_argument("--uv-cache-dir", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workdir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.hermes_bin.is_file():
        raise SystemExit(f"Hermes binary not found: {args.hermes_bin}")
    if not args.prompt_file.is_file():
        raise SystemExit(f"Prompt file not found: {args.prompt_file}")
    if not args.workdir.is_dir():
        raise SystemExit(f"Workdir not found: {args.workdir}")

    prompt = args.prompt_file.read_text(encoding="utf-8")
    env = os.environ.copy()
    env["HERMES_HOME"] = str(args.hermes_home)
    env["UV_CACHE_DIR"] = str(args.uv_cache_dir)

    command = [
        str(args.hermes_bin),
        "--ignore-rules",
        "--ignore-user-config",
        "--provider",
        args.provider,
        "-m",
        args.model,
        "-z",
        prompt,
    ]
    # Intentional benchmark runner: callers pass the exact isolated Hermes
    # binary to execute, and we avoid shell=True so prompt text cannot affect
    # process parsing.
    result = subprocess.run(  # noqa: S603
        command,
        cwd=args.workdir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
