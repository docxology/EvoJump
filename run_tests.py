#!/usr/bin/env python3
"""Thin orchestrator over the pytest suite.

Delegates entirely to pytest (discovery, selection, reporting); this wrapper
only chooses a sensible default invocation and surfaces pass/fail + exit code.
The canonical direct command is:
    MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q --no-cov -p no:cacheprovider

Usage:
    python run_tests.py [--quick] [--coverage] [--fail-under N] [pytest args...]
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FAIL_UNDER = 68  # mirrors pyproject addopts --cov-fail-under=68


def build_cmd(argv: list[str]) -> list[str]:
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    args = list(argv)
    coverage = False
    if "--coverage" in args:
        coverage = True
        args.remove("--coverage")
    if "--quick" in args:
        args.remove("--quick")
        coverage = False
        if "--no-cov" not in args:
            args.insert(0, "--no-cov")
    if not coverage and "--no-cov" not in args and not any(
        a.startswith("--cov") for a in args
    ):
        # pyproject addopts always inject --cov*; disable unless asked for.
        args.insert(0, "--no-cov")
    if not any(a.startswith("--cov-fail-under") for a in args) and coverage:
        args.append(f"--cov-fail-under={DEFAULT_FAIL_UNDER}")
    return cmd + args


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = build_cmd(argv)
    print("Running:", " ".join(cmd), flush=True)
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return proc.returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
