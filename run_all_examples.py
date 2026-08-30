#!/usr/bin/env python3
"""Thin orchestrator: discover and run every example under examples/.

Discovers example files at runtime (no hardcoded roster), runs each in a
subprocess with a timeout, and aggregates pass/fail. Business logic lives in
src/evojump/; scripts that are workaround-for-bug relics live in
examples/archive/ and are skipped automatically.

Usage:
    python run_all_examples.py [--timeout N] [--verbose] [--fail-fast]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
ARCHIVE_DIR = EXAMPLES_DIR / "archive"
DEFAULT_TIMEOUT = 900  # seconds; generous because examples can render videos


def discover_examples() -> list[Path]:
    """Return runnable example scripts (top level only, archive excluded)."""
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(
        p for p in EXAMPLES_DIR.glob("*.py")
        if p.name != "__init__.py"
    )


def run_example(path: Path, timeout: int) -> tuple[bool, float, str]:
    """Run one example in a subprocess; return (success, elapsed, tail)."""
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start
        ok = proc.returncode == 0
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        return ok, elapsed, tail
    except subprocess.TimeoutExpired:
        return False, time.time() - start, f"TIMEOUT after {timeout}s"
    except Exception as exc:  # noqa: BLE001 - orchestrator must survive crashes
        return False, time.time() - start, f"LAUNCH ERROR: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all EvoJump examples")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="Per-example timeout in seconds")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    examples = discover_examples()
    if not examples:
        print(f"No examples found in {EXAMPLES_DIR}")
        return 1
    if ARCHIVE_DIR.is_dir():
        print(f"(skipping archived workaround scripts in examples/archive/)")

    results: list[tuple[str, bool, float, str]] = []
    failures = 0
    for path in examples:
        print(f"Running {path.name} ...", flush=True)
        ok, elapsed, tail = run_example(path, args.timeout)
        status = "PASS" if ok else "FAIL"
        print(f"  {status} ({elapsed:.1f}s)")
        results.append((path.name, ok, elapsed, tail))
        if not ok:
            failures += 1
            if args.verbose and tail:
                print("  --- output tail ---")
                for line in tail.strip().splitlines()[-15:]:
                    print(f"  {line}")
            if args.fail_fast:
                break

    print("\n=== Summary ===")
    for name, ok, elapsed, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name} ({elapsed:.1f}s)")
    print(f"{len(results) - failures}/{len(results)} examples passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
