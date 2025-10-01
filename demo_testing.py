#!/usr/bin/env python3
"""
EvoJump Testing Framework Demonstration

This script demonstrates the comprehensive testing framework for EvoJump,
showing how to run different types of tests, coverage reports, and validation checks.

Usage:
    python demo_testing.py [demo_type]

Demo Types:
    quick       - Demonstrate quick testing (default)
    coverage    - Demonstrate coverage testing
    benchmark   - Demonstrate performance benchmarking
    all         - Run all demonstration types
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and report the result."""
    print(f"\n🔍 {description}")
    print(f"   Command: {' '.join(cmd)}")

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()

        print(f"   ✅ Success! (took {end_time - start_time:.2f}s)")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()[:200]}...")
        return True

    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed! (took {end_time - start_time:.2f}s)")
        if e.stderr.strip():
            print(f"   Error: {e.stderr.strip()[:200]}...")
        return False


def demo_quick_testing():
    """Demonstrate quick testing functionality."""
    print("🚀 DEMO: Quick Testing")
    print("=" * 50)

    success = run_command(
        ["python", "run_all_tests.py", "--quick", "--verbose"],
        "Running quick tests (fast feedback)"
    )
    return success


def demo_coverage_testing():
    """Demonstrate coverage testing functionality."""
    print("🚀 DEMO: Coverage Testing")
    print("=" * 50)

    success = run_command(
        ["python", "run_all_tests.py", "--coverage"],
        "Running tests with comprehensive coverage report"
    )

    # Check if coverage report was generated
    if (Path("htmlcov") / "index.html").exists():
        print("   📊 Coverage report available at: htmlcov/index.html")
    else:
        print("   ⚠️  Coverage report not found")

    return success


def demo_benchmark_testing():
    """Demonstrate benchmark testing functionality."""
    print("🚀 DEMO: Performance Benchmarking")
    print("=" * 50)

    success = run_command(
        ["python", "run_all_tests.py", "--benchmark", "--parallel"],
        "Running performance benchmarks in parallel"
    )
    return success


def demo_code_quality():
    """Demonstrate code quality checks."""
    print("🚀 DEMO: Code Quality Checks")
    print("=" * 50)

    # Check if quality tools are available
    tools = ["black", "flake8", "mypy"]
    available_tools = []

    for tool in tools:
        if subprocess.run(["which", tool], capture_output=True).returncode == 0:
            available_tools.append(tool)

    if not available_tools:
        print("   ⚠️  No code quality tools found. Install with: pip install black flake8 mypy")
        return False

    success = True

    for tool in available_tools:
        cmd = [tool, "--help"]
        success &= run_command(cmd, f"Checking {tool} availability")

    return success


def demo_documentation():
    """Demonstrate documentation checks."""
    print("🚀 DEMO: Documentation Checks")
    print("=" * 50)

    success = run_command(
        ["python", "run_all_tests.py", "--docs"],
        "Checking documentation completeness"
    )
    return success


def demo_all():
    """Run all demonstrations."""
    print("🚀 DEMO: Complete Testing Framework")
    print("=" * 60)

    demos = [
        demo_quick_testing,
        demo_coverage_testing,
        demo_benchmark_testing,
        demo_code_quality,
        demo_documentation
    ]

    results = []
    for demo_func in demos:
        try:
            results.append(demo_func())
        except Exception as e:
            print(f"   💥 Demo failed: {e}")
            results.append(False)

    successful = sum(results)
    total = len(results)

    print("\n🎯 DEMO SUMMARY")
    print("=" * 60)
    print(f"📊 Results: {successful}/{total} demonstrations successful")

    if successful == total:
        print("🎉 All demonstrations completed successfully!")
        print("🚀 EvoJump testing framework is working perfectly!")
    else:
        print("⚠️  Some demonstrations had issues.")
        print("🔧 Check the error messages above for details.")

    return successful == total


def main():
    """Main demonstration entry point."""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print(__doc__)
        return 0

    if len(sys.argv) > 1:
        demo_type = sys.argv[1].lower()
    else:
        demo_type = "quick"

    print("EvoJump Testing Framework Demonstration")
    print("=" * 60)

    if demo_type == "quick":
        success = demo_quick_testing()
    elif demo_type == "coverage":
        success = demo_coverage_testing()
    elif demo_type == "benchmark":
        success = demo_benchmark_testing()
    elif demo_type == "quality":
        success = demo_code_quality()
    elif demo_type == "docs":
        success = demo_documentation()
    elif demo_type == "all":
        success = demo_all()
    else:
        print(f"❌ Unknown demo type: {demo_type}")
        print("Available demo types: quick, coverage, benchmark, quality, docs, all")
        return 1

    print("\n📚 Available Commands:")
    print("   python run_all_tests.py --quick      # Fast feedback")
    print("   python run_all_tests.py --coverage   # Full coverage report")
    print("   python run_all_tests.py --benchmark  # Performance tests")
    print("   python run_all_tests.py --lint       # Code quality checks")
    print("   python run_all_tests.py --docs       # Documentation checks")
    print("   python run_all_tests.py --all        # Complete validation")
    print("   pytest                               # Standard pytest commands")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
