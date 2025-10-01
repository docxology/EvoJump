#!/usr/bin/env python3
"""
EvoJump Complete Test Suite Runner

This script runs the complete test suite for the EvoJump project, providing
comprehensive testing, coverage reporting, and performance benchmarks.

Usage:
    python run_all_tests.py [options]

Options:
    --quick, -q         - Run tests quickly without coverage (fast feedback)
    --coverage, -c      - Run tests with detailed coverage report
    --verbose, -v       - Verbose output
    --parallel, -p      - Run tests in parallel
    --benchmark, -b     - Run performance benchmarks
    --lint, -l          - Run code quality checks (black, flake8, mypy)
    --docs, -d          - Check documentation completeness
    --all, -a           - Run all checks (tests + lint + docs)
    --fail-fast         - Stop on first test failure
    --help, -h          - Show this help message

Examples:
    python run_all_tests.py --quick                    # Fast feedback
    python run_all_tests.py --coverage --verbose       # Full coverage report
    python run_all_tests.py --benchmark --parallel     # Performance tests
    python run_all_tests.py --all                      # Complete validation
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
import platform
import shutil
from typing import List, Dict, Any


class CompleteTestRunner:
    """Comprehensive test runner for EvoJump project."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
        self.src_dir = self.project_root / "src"
        self.docs_dir = self.project_root / "docs"

        # Test configuration
        self.test_modules = [
            "test_datacore",
            "test_jumprope",
            "test_laserplane",
            "test_trajectory_visualizer",
            "test_analytics_engine",
            "test_evolution_sampler",
            "test_advanced_features",
            "test_cli"
        ]

        self.coverage_threshold = 95
        self.benchmark_tests = [
            "test_jumprope::TestOrnsteinUhlenbeckJump::test_simulate_trajectory",
            "test_laserplane::TestDistributionFitter::test_fit_distribution_normal",
            "test_analytics_engine::TestTimeSeriesAnalyzer::test_analyze_trends_linear"
        ]

    def run_command(self, cmd: List[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        try:
            return subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                check=check
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {' '.join(cmd)}")
            if capture_output and e.stderr:
                print(f"   Error: {e.stderr}")
            return e

    def print_header(self, title: str):
        """Print a formatted header."""
        print(f"\n{'='*60}")
        print(f"🚀 {title}")
        print('='*60)

    def run_quick_tests(self, args: argparse.Namespace) -> bool:
        """Run tests quickly without coverage."""
        self.print_header("QUICK TESTS")

        cmd = [
            "python", "-m", "pytest",
            "--tb=short",
            "--durations=10"
        ]

        if args.verbose:
            cmd.append("-v")

        if args.parallel:
            cmd.extend(["-n", "auto"])

        if args.fail_fast:
            cmd.append("--maxfail=1")

        result = self.run_command(cmd)
        return result.returncode == 0

    def run_full_coverage_tests(self, args: argparse.Namespace) -> bool:
        """Run tests with comprehensive coverage."""
        self.print_header("FULL COVERAGE TESTS")

        cmd = [
            "python", "-m", "pytest",
            f"--cov=evojump",
            "--cov-report=term-missing",
            f"--cov-fail-under={self.coverage_threshold}",
            "--cov-report=html:htmlcov",
            "--cov-report=xml",
            "--tb=short",
            "--durations=10"
        ]

        if args.verbose:
            cmd.append("-v")

        if args.parallel:
            cmd.extend(["-n", "auto"])

        if args.fail_fast:
            cmd.append("--maxfail=1")

        result = self.run_command(cmd)
        return result.returncode == 0

    def run_specific_test_modules(self, args: argparse.Namespace) -> bool:
        """Run specific test modules."""
        self.print_header("SPECIFIC TEST MODULES")

        all_passed = True

        for module in self.test_modules:
            print(f"\n🔍 Running {module}.py...")

            cmd = [
                "python", "-m", "pytest",
                f"tests/{module}.py",
                "--tb=short",
                "--durations=5"
            ]

            if args.verbose:
                cmd.append("-v")

            result = self.run_command(cmd, check=False)

            if result.returncode != 0:
                all_passed = False
                print(f"❌ {module}.py failed!")

            # Show summary for each module
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:  # Show last 5 lines
                    if any(keyword in line for keyword in ['passed', 'failed', 'errors', 'warnings']):
                        print(f"   {line}")

        return all_passed

    def run_benchmark_tests(self, args: argparse.Namespace) -> bool:
        """Run performance benchmarks."""
        self.print_header("PERFORMANCE BENCHMARKS")

        cmd = [
            "python", "-m", "pytest",
            "-k", "benchmark or performance or simulate or fit",
            "--tb=short",
            "--durations=0",
            "--benchmark-only"
        ]

        if args.verbose:
            cmd.append("-v")

        if args.parallel:
            cmd.extend(["-n", "auto"])

        result = self.run_command(cmd, check=False)

        # Check if benchmarks ran successfully
        if "benchmark" in result.stdout.lower() or result.returncode == 0:
            print("✅ Performance benchmarks completed!")
            return True
        else:
            print("⚠️  No benchmarks found or benchmarks failed!")
            return False

    def run_linting_checks(self, args: argparse.Namespace) -> bool:
        """Run code quality checks."""
        self.print_header("CODE QUALITY CHECKS")

        all_passed = True

        # Check if black is available
        if shutil.which("black"):
            print("🔧 Running Black (code formatting)...")
            result = self.run_command(["black", "--check", "--diff", "src/", "tests/"], check=False)
            if result.returncode != 0:
                all_passed = False
                print("❌ Black formatting issues found!")
            else:
                print("✅ Black formatting OK!")

        # Check if flake8 is available
        if shutil.which("flake8"):
            print("🔧 Running Flake8 (style guide)...")
            result = self.run_command(["flake8", "src/", "tests/"], check=False)
            if result.returncode != 0:
                all_passed = False
                print("❌ Flake8 style issues found!")
            else:
                print("✅ Flake8 style OK!")

        # Check if mypy is available
        if shutil.which("mypy"):
            print("🔧 Running MyPy (type checking)...")
            result = self.run_command(["mypy", "src/", "tests/"], check=False)
            if result.returncode != 0:
                all_passed = False
                print("❌ MyPy type issues found!")
            else:
                print("✅ MyPy types OK!")

        return all_passed

    def check_documentation(self, args: argparse.Namespace) -> bool:
        """Check documentation completeness."""
        self.print_header("DOCUMENTATION CHECKS")

        all_passed = True

        # Check if Sphinx documentation builds
        if (self.docs_dir / "conf.py").exists():
            print("📚 Building Sphinx documentation...")
            result = self.run_command([
                "python", "-m", "sphinx",
                "-b", "html",
                "-D", "extensions=sphinx.ext.autodoc",
                "docs/", "docs/_build/html"
            ], check=False)

            if result.returncode != 0:
                all_passed = False
                print("❌ Sphinx documentation build failed!")
            else:
                print("✅ Sphinx documentation OK!")

        # Check module docstrings
        print("📋 Checking module docstrings...")
        missing_docs = []

        for py_file in self.src_dir.rglob("*.py"):
            if py_file.name.startswith("__") or py_file.name.startswith("test"):
                continue

            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if not content.startswith('"""') or '"""' not in content[:200]:
                        missing_docs.append(py_file.name)
            except:
                pass

        if missing_docs:
            all_passed = False
            print(f"❌ Missing module docstrings: {', '.join(missing_docs)}")
        else:
            print("✅ All modules have docstrings!")

        return all_passed

    def run_comprehensive_validation(self, args: argparse.Namespace) -> bool:
        """Run all validation checks."""
        self.print_header("COMPREHENSIVE VALIDATION")

        print("🎯 Running complete EvoJump validation suite...")
        print(f"   Python {platform.python_version()} on {platform.system()}")

        checks = [
            ("Quick Tests", self.run_quick_tests, True),
            ("Coverage Tests", self.run_full_coverage_tests, True),
            ("Specific Modules", self.run_specific_test_modules, True),
            ("Performance Benchmarks", self.run_benchmark_tests, False),  # Optional
            ("Code Quality", self.run_linting_checks, False),  # Optional
            ("Documentation", self.check_documentation, False),  # Optional
        ]

        results = {}

        for check_name, check_func, required in checks:
            print(f"\n{'='*40}")
            print(f"Running {check_name}...")
            print('='*40)

            try:
                success = check_func(args)
                results[check_name] = success

                if success:
                    print(f"✅ {check_name} PASSED")
                else:
                    print(f"❌ {check_name} FAILED")

                if not success and required:
                    print(f"🚨 {check_name} is required but failed!")
                    break

            except Exception as e:
                print(f"💥 {check_name} crashed: {str(e)}")
                results[check_name] = False
                if required:
                    break

        # Summary
        self.print_header("VALIDATION SUMMARY")

        passed = sum(1 for result in results.values() if result)
        total = len(results)

        print(f"📊 Results: {passed}/{total} checks passed")

        if passed == total:
            print("🎉 ALL VALIDATION CHECKS PASSED!")
            print("🚀 EvoJump is ready for production!")
        else:
            print("⚠️  Some validation checks failed.")
            print("🔧 Please fix the issues before proceeding.")

        # Detailed breakdown
        print("\n📋 Detailed Results:")
        for check_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status}: {check_name}")

        return passed == total

    def run(self, args: argparse.Namespace) -> bool:
        """Main execution method."""
        start_time = time.time()

        print("EvoJump Complete Test Suite Runner")
        print("=" * 50)

        if args.all:
            return self.run_comprehensive_validation(args)

        # Run specific checks based on arguments
        if args.quick:
            success = self.run_quick_tests(args)
        elif args.coverage:
            success = self.run_full_coverage_tests(args)
        elif args.benchmark:
            success = self.run_benchmark_tests(args)
        elif args.lint:
            success = self.run_linting_checks(args)
        elif args.docs:
            success = self.check_documentation(args)
        else:
            # Default to comprehensive validation
            success = self.run_comprehensive_validation(args)

        end_time = time.time()
        print(f"\n⏱️  Total execution time: {end_time - start_time:.2f}s")

        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EvoJump Complete Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Run tests quickly without coverage"
    )

    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Run tests with detailed coverage report"
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
        "-b", "--benchmark",
        action="store_true",
        help="Run performance benchmarks"
    )

    parser.add_argument(
        "-l", "--lint",
        action="store_true",
        help="Run code quality checks"
    )

    parser.add_argument(
        "-d", "--docs",
        action="store_true",
        help="Check documentation completeness"
    )

    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Run all checks (tests + lint + docs)"
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first test failure"
    )

    args = parser.parse_args()

    # Initialize and run tests
    runner = CompleteTestRunner()
    success = runner.run(args)

    # Exit with appropriate code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

