#!/usr/bin/env python3
"""
EvoJump Test Runner

A comprehensive test execution framework for the EvoJump project that provides
multiple testing modes, coverage reporting, and performance benchmarking.

Usage:
    python run_tests.py [mode] [options]

Modes:
    quick       - Run tests quickly without coverage (fast feedback)
    full        - Run all tests with coverage (default)
    coverage    - Run tests with detailed coverage report
    benchmark   - Run performance benchmarks
    specific    - Run specific test files or classes
    integration - Run integration tests only
    unit        - Run unit tests only
    ci          - Run tests in CI mode (strict, no coverage)

Options:
    --verbose, -v      - Verbose output
    --parallel, -p     - Run tests in parallel
    --no-coverage      - Skip coverage reporting
    --html             - Generate HTML coverage report
    --xml              - Generate XML coverage report
    --fail-under=N     - Set minimum coverage threshold (default: 95)
    --profile          - Profile test execution time
    --memory           - Monitor memory usage
    --help, -h         - Show this help message

Examples:
    python run_tests.py quick
    python run_tests.py full --verbose --html
    python run_tests.py specific test_datacore.py
    python run_tests.py benchmark --profile
    python run_tests.py ci --fail-under=90
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
import psutil
import cProfile
import pstats
from typing import List, Optional, Dict, Any


class TestRunner:
    """Comprehensive test runner for EvoJump project."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
        self.htmlcov_dir = self.project_root / "htmlcov"
        self.coverage_file = self.project_root / "coverage.xml"

        # Test configuration
        self.test_paths = ["tests"]
        self.python_files = ["test_*.py", "*_test.py"]
        self.python_classes = ["Test*"]
        self.python_functions = ["test_*"]

        # Default coverage settings
        self.default_fail_under = 95

    def run_command(self, cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        try:
            return subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            return e

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }

    def run_quick_tests(self, args: argparse.Namespace) -> bool:
        """Run tests quickly without coverage for fast feedback."""
        print("🧪 Running quick tests (no coverage)...")

        cmd = [
            "python", "-m", "pytest",
            "--tb=short",
            "--durations=10"
        ]

        if args.verbose:
            cmd.append("-v")

        if args.parallel:
            cmd.extend(["-n", "auto"])

        # Run tests
        start_time = time.time()
        start_memory = self.get_memory_usage()

        result = self.run_command(cmd)

        end_time = time.time()
        end_memory = self.get_memory_usage()

        # Report results
        if result.returncode == 0:
            print("✅ Quick tests passed!")
            print(f"⏱️  Execution time: {end_time - start_time:.2f}s")
            if args.memory:
                print(f"💾 Memory usage: {end_memory['rss_mb'] - start_memory['rss_mb']:.1f}MB")
            return True
        else:
            print("❌ Quick tests failed!")
            return False

    def run_full_tests(self, args: argparse.Namespace) -> bool:
        """Run all tests with comprehensive coverage."""
        print("🧪 Running full test suite with coverage...")

        cmd = [
            "python", "-m", "pytest",
            f"--cov=evojump",
            "--cov-report=term-missing",
            f"--cov-fail-under={args.fail_under}",
            "--tb=short",
            "--durations=10"
        ]

        if args.verbose:
            cmd.append("-v")

        if args.parallel:
            cmd.extend(["-n", "auto"])

        if args.html:
            cmd.extend(["--cov-report=html:htmlcov"])

        if args.xml:
            cmd.extend(["--cov-report=xml"])

        # Run tests
        start_time = time.time()
        start_memory = self.get_memory_usage()

        result = self.run_command(cmd)

        end_time = time.time()
        end_memory = self.get_memory_usage()

        # Report results
        if result.returncode == 0:
            print("✅ Full tests passed!")
            print(f"⏱️  Execution time: {end_time - start_time:.2f}s")
            if args.memory:
                print(f"💾 Memory usage: {end_memory['rss_mb'] - start_memory['rss_mb']:.1f}MB")
            return True
        else:
            print("❌ Full tests failed!")
            return False

    def run_specific_tests(self, args: argparse.Namespace) -> bool:
        """Run specific test files or classes."""
        if not args.targets:
            print("❌ Error: No test targets specified for 'specific' mode")
            return False

        print(f"🧪 Running specific tests: {', '.join(args.targets)}")

        cmd = [
            "python", "-m", "pytest",
            "--tb=short"
        ] + args.targets

        if args.verbose:
            cmd.append("-v")

        if args.parallel:
            cmd.extend(["-n", "auto"])

        result = self.run_command(cmd)

        if result.returncode == 0:
            print("✅ Specific tests passed!")
            return True
        else:
            print("❌ Specific tests failed!")
            return False

    def run_benchmark_tests(self, args: argparse.Namespace) -> bool:
        """Run performance benchmarks."""
        print("🏃 Running performance benchmarks...")

        benchmark_tests = [
            "tests/test_performance_benchmarks.py",
            "tests/test_jumprope.py::TestOrnsteinUhlenbeckJump::test_simulate_trajectory",
            "tests/test_laserplane.py::TestDistributionFitter::test_fit_distribution_normal",
            "tests/test_analytics_engine.py::TestTimeSeriesAnalyzer::test_analyze_trends_linear"
        ]

        cmd = [
            "python", "-m", "pytest",
            "--tb=short",
            "--durations=0",  # Show all durations
            "-k", "benchmark or performance or simulate or fit"
        ]

        if args.profile:
            cmd.extend(["--profile"])

        if args.verbose:
            cmd.append("-v")

        # Run benchmarks
        start_time = time.time()
        start_memory = self.get_memory_usage()

        result = self.run_command(cmd)

        end_time = time.time()
        end_memory = self.get_memory_usage()

        # Report results
        if result.returncode == 0:
            print("✅ Benchmark tests completed!")
            print(f"⏱️  Total execution time: {end_time - start_time:.2f}s")
            if args.memory:
                print(f"💾 Memory usage: {end_memory['rss_mb'] - start_memory['rss_mb']:.1f}MB")
            return True
        else:
            print("❌ Benchmark tests failed!")
            return False

    def run_ci_tests(self, args: argparse.Namespace) -> bool:
        """Run tests in CI mode (strict, no coverage)."""
        print("🔄 Running tests in CI mode...")

        cmd = [
            "python", "-m", "pytest",
            "--tb=short",
            "--strict-markers",
            "--strict-config",
            f"--cov-fail-under={args.fail_under}",
            "--durations=10"
        ]

        # CI mode is strict - no coverage by default unless specified
        if not args.no_coverage:
            cmd.extend([
                "--cov=evojump",
                "--cov-report=term-missing"
            ])

        if args.verbose:
            cmd.append("-v")

        result = self.run_command(cmd)

        if result.returncode == 0:
            print("✅ CI tests passed!")
            return True
        else:
            print("❌ CI tests failed!")
            return False

    def run_integration_tests(self, args: argparse.Namespace) -> bool:
        """Run only integration tests."""
        print("🔗 Running integration tests...")

        cmd = [
            "python", "-m", "pytest",
            "-k", "integration or fit or analyze or compare",
            "--tb=short"
        ]

        if args.verbose:
            cmd.append("-v")

        if args.parallel:
            cmd.extend(["-n", "auto"])

        result = self.run_command(cmd)

        if result.returncode == 0:
            print("✅ Integration tests passed!")
            return True
        else:
            print("❌ Integration tests failed!")
            return False

    def run_unit_tests(self, args: argparse.Namespace) -> bool:
        """Run only unit tests."""
        print("🧩 Running unit tests...")

        cmd = [
            "python", "-m", "pytest",
            "-k", "not integration and not fit and not analyze and not compare",
            "--tb=short"
        ]

        if args.verbose:
            cmd.append("-v")

        if args.parallel:
            cmd.extend(["-n", "auto"])

        result = self.run_command(cmd)

        if result.returncode == 0:
            print("✅ Unit tests passed!")
            return True
        else:
            print("❌ Unit tests failed!")
            return False

    def show_help(self):
        """Display help information."""
        print(__doc__)

    def run(self, args: argparse.Namespace) -> bool:
        """Main test execution method."""
        print(f"EvoJump Test Runner - {args.mode.upper()} Mode")
        print("=" * 50)

        # Execute appropriate test mode
        if args.mode == "quick":
            return self.run_quick_tests(args)
        elif args.mode == "full":
            return self.run_full_tests(args)
        elif args.mode == "coverage":
            # Override to ensure coverage is run
            args.html = True
            args.xml = True
            return self.run_full_tests(args)
        elif args.mode == "benchmark":
            return self.run_benchmark_tests(args)
        elif args.mode == "specific":
            return self.run_specific_tests(args)
        elif args.mode == "integration":
            return self.run_integration_tests(args)
        elif args.mode == "unit":
            return self.run_unit_tests(args)
        elif args.mode == "ci":
            return self.run_ci_tests(args)
        else:
            print(f"❌ Unknown mode: {args.mode}")
            self.show_help()
            return False


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="EvoJump Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py quick                    # Fast feedback
  python run_tests.py full --verbose --html    # Full coverage report
  python run_tests.py specific test_datacore.py  # Specific tests
  python run_tests.py benchmark --profile      # Performance tests
  python run_tests.py ci --fail-under=90       # CI mode
        """
    )

    parser.add_argument(
        "mode",
        nargs="?",
        default="full",
        choices=["quick", "full", "coverage", "benchmark", "specific", "integration", "unit", "ci"],
        help="Test execution mode"
    )

    parser.add_argument(
        "targets",
        nargs="*",
        help="Specific test targets (for 'specific' mode)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )

    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage reporting"
    )

    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report"
    )

    parser.add_argument(
        "--xml",
        action="store_true",
        help="Generate XML coverage report"
    )

    parser.add_argument(
        "--fail-under",
        type=int,
        default=95,
        help="Minimum coverage threshold (default: 95)"
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile test execution time"
    )

    parser.add_argument(
        "--memory",
        action="store_true",
        help="Monitor memory usage"
    )


    args = parser.parse_args()

    # Initialize and run tests
    runner = TestRunner()
    success = runner.run(args)

    # Exit with appropriate code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
