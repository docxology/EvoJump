#!/usr/bin/env python3
"""Thin orchestrator: run the pytest suite (tests/) and report.

Legacy variant of run_tests.py kept as a stable entry point; delegates to the
same pytest invocation. Extra args are forwarded to pytest verbatim.

Usage:
    python run_all_tests.py [--quick] [pytest args...]
"""
import sys

from run_tests import main  # re-export the single orchestrator implementation

if __name__ == "__main__":
    sys.exit(main())
