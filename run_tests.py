#!/usr/bin/env python3
"""Thin orchestrator over the pytest suite.

Delegates entirely to pytest (discovery, selection, reporting); this wrapper
only chooses a sensible default invocation and surfaces pass/fail + exit code.
The canonical direct command is:
    MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q --no-cov -p no:cacheprovider

Coverage gate: direct `coverage run` + `coverage report --fail-under` over
src/evojump (pytest-cov args crash on numpy>=2.5 — see the pyproject note).

Usage:
    python run_tests.py [--quick] [--coverage] [pytest args...]
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FAIL_UNDER = 95  # coverage report gate over src/evojump (see pyproject note)


def build_cmds(argv: list[str]) -> list[list[str]]:
    """Return the command(s) to run: pytest, or coverage run + coverage report."""
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
    if coverage:
        run_cmd = COVERAGE_RUN + ["-m", "pytest", "tests/"] + args
        report_cmd = [
            ".venv/bin/coverage",
            "report",
            f"--fail-under={DEFAULT_FAIL_UNDER}",
        ]
        return [run_cmd, report_cmd]
    cmd = [str(PROJECT_ROOT / ".venv/bin/python"), "-m", "pytest", "tests/"]
    if "--no-cov" not in args and not any(a.startswith("--cov") for a in args):
        # pyproject addopts no longer inject --cov*; keep plain runs fast anyway.
        args.insert(0, "--no-cov")
    return [cmd + args]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    for cmd in build_cmds(argv):
        print("Running:", " ".join(cmd), flush=True)
        try:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        except KeyboardInterrupt:
            return 130
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
