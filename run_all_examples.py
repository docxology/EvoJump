#!/usr/bin/env python3
"""
Comprehensive EvoJump Examples Validation

This script runs and validates all EvoJump examples to ensure they are working correctly
and demonstrate the full capabilities of the framework.

Usage:
    python run_all_examples.py [options]

Options:
    --quick, -q         - Run quick validation (skip slow examples)
    --verbose, -v       - Verbose output
    --parallel, -p      - Run tests in parallel where possible
    --fail-fast         - Stop on first failure
    --examples-only     - Only run examples, skip testing
    --tests-only        - Only run tests, skip examples
    --help, -h          - Show this help message

Examples:
    python run_all_examples.py --quick                    # Fast validation
    python run_all_examples.py --verbose --parallel      # Full validation
    python run_all_examples.py --fail-fast               # Stop on errors
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
import importlib.util
from typing import List, Dict, Any, Tuple


class ExamplesValidator:
    """Comprehensive validator for all EvoJump examples."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.examples_dir = self.project_root / "examples"

        # Define all examples with their properties
        self.examples = {
            'basic_usage_fixed': {
                'file': 'basic_usage_fixed.py',
                'description': 'Basic workflow from data loading to visualization',
                'category': 'basic',
                'estimated_time': 30,  # seconds
                'dependencies': ['matplotlib', 'pandas', 'numpy']
            },
            'working_demo': {
                'file': 'working_demo.py',
                'description': 'Working demonstration of all core features',
                'category': 'basic',
                'estimated_time': 45,
                'dependencies': ['matplotlib', 'pandas', 'numpy']
            },
            'comprehensive_demo': {
                'file': 'comprehensive_demo.py',
                'description': 'Full analysis pipeline with all modules',
                'category': 'advanced',
                'estimated_time': 90,
                'dependencies': ['matplotlib', 'pandas', 'numpy']
            },
            'comprehensive_advanced_analytics_demo': {
                'file': 'comprehensive_advanced_analytics_demo.py',
                'description': 'Advanced statistical methods demonstration',
                'category': 'advanced',
                'estimated_time': 120,
                'dependencies': ['matplotlib', 'pandas', 'numpy', 'scipy']
            },
            'advanced_features_demo': {
                'file': 'advanced_features_demo.py',
                'description': 'Advanced stochastic process models',
                'category': 'advanced',
                'estimated_time': 180,
                'dependencies': ['matplotlib', 'pandas', 'numpy']
            },
            'animation_demo': {
                'file': 'animation_demo.py',
                'description': 'Basic animation generation',
                'category': 'visualization',
                'estimated_time': 60,
                'dependencies': ['matplotlib', 'pandas', 'numpy']
            },
            'enhanced_animation_demo': {
                'file': 'enhanced_animation_demo.py',
                'description': 'Multi-condition and comparative animations',
                'category': 'visualization',
                'estimated_time': 120,
                'dependencies': ['matplotlib', 'pandas', 'numpy']
            },
            'comprehensive_animation_demo': {
                'file': 'comprehensive_animation_demo.py',
                'description': 'Advanced animation with multiple types',
                'category': 'visualization',
                'estimated_time': 180,
                'dependencies': ['matplotlib', 'pandas', 'numpy']
            },
            'simple_animation_demo': {
                'file': 'simple_animation_demo.py',
                'description': 'Simple trajectory animation',
                'category': 'visualization',
                'estimated_time': 30,
                'dependencies': ['matplotlib', 'pandas', 'numpy']
            },
            'thin_orchestrator_examples': {
                'file': 'thin_orchestrator_examples.py',
                'description': 'Thin orchestrator patterns',
                'category': 'architecture',
                'estimated_time': 60,
                'dependencies': ['pandas', 'numpy']
            },
            'thin_orchestrator_working': {
                'file': 'thin_orchestrator_working.py',
                'description': 'Working thin orchestrator implementation',
                'category': 'architecture',
                'estimated_time': 60,
                'dependencies': ['pandas', 'numpy']
            },
            'performance_benchmarks': {
                'file': 'performance_benchmarks.py',
                'description': 'Performance testing and benchmarking',
                'category': 'performance',
                'estimated_time': 120,
                'dependencies': ['matplotlib', 'pandas', 'numpy']
            },
            'drosophila_case_study': {
                'file': 'drosophila_case_study.py',
                'description': 'Complete fruit fly biology analysis',
                'category': 'case_study',
                'estimated_time': 300,  # Can be long due to comprehensive analysis
                'dependencies': ['matplotlib', 'pandas', 'numpy', 'scipy']
            }
        }

        self.categories = {
            'basic': '🚀 Basic Usage Examples',
            'advanced': '📊 Advanced Analytics Examples',
            'visualization': '🎨 Visualization Examples',
            'architecture': '🏗️ Architecture Examples',
            'performance': '⚡ Performance Examples',
            'case_study': '🧬 Case Study Examples'
        }

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

    def check_dependencies(self, example_name: str) -> bool:
        """Check if required dependencies are available for an example."""
        example = self.examples[example_name]

        for dep in example['dependencies']:
            try:
                importlib.import_module(dep)
            except ImportError:
                print(f"⚠️  Missing dependency for {example_name}: {dep}")
                return False

        return True

    def run_single_example(self, example_name: str, args: argparse.Namespace) -> Tuple[bool, float]:
        """Run a single example and return success status and execution time."""
        example = self.examples[example_name]
        example_file = self.examples_dir / example['file']

        if not example_file.exists():
            print(f"❌ Example file not found: {example_file}")
            return False, 0.0

        if not self.check_dependencies(example_name):
            print(f"❌ Dependencies not satisfied for {example_name}")
            return False, 0.0

        print(f"🔍 Running {example_name}: {example['description']}")

        start_time = time.time()

        try:
            # Run the example
            result = self.run_command(
                [sys.executable, str(example_file)],
                capture_output=not args.verbose,
                check=False
            )

            end_time = time.time()
            execution_time = end_time - start_time

            if result.returncode == 0:
                print(f"✅ {example_name} completed successfully ({execution_time:.1f}s)")
                return True, execution_time
            else:
                print(f"❌ {example_name} failed ({execution_time:.1f}s)")
                if args.verbose and result.stderr:
                    print(f"   Error: {result.stderr}")
                return False, execution_time

        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"💥 {example_name} crashed ({execution_time:.1f}s): {e}")
            return False, execution_time

    def run_examples_by_category(self, category: str, args: argparse.Namespace) -> Dict[str, Tuple[bool, float]]:
        """Run all examples in a specific category."""
        results = {}

        for example_name, example_info in self.examples.items():
            if example_info['category'] == category:
                success, exec_time = self.run_single_example(example_name, args)
                results[example_name] = (success, exec_time)

        return results

    def run_all_examples(self, args: argparse.Namespace) -> Dict[str, Tuple[bool, float]]:
        """Run all examples."""
        all_results = {}

        print("🚀 EvoJump Examples Validation")
        print("=" * 60)

        for category_name, category_description in self.categories.items():
            if args.quick and category_name in ['advanced', 'performance', 'case_study']:
                print(f"⏭️  Skipping {category_description} (quick mode)")
                continue

            print(f"\n{category_description}")
            print("-" * 40)

            category_results = self.run_examples_by_category(category_name, args)

            for example_name, (success, exec_time) in category_results.items():
                all_results[example_name] = (success, exec_time)

        return all_results

    def run_tests(self, args: argparse.Namespace) -> bool:
        """Run the test suite."""
        print("🧪 Running EvoJump Test Suite")
        print("-" * 40)

        start_time = time.time()

        # Run tests
        cmd = [sys.executable, "-m", "pytest", "tests/"]

        if args.verbose:
            cmd.append("-v")

        if args.parallel:
            cmd.extend(["-n", "auto"])

        if args.fail_fast:
            cmd.append("--maxfail=1")

        result = self.run_command(cmd, check=False)

        end_time = time.time()
        execution_time = end_time - start_time

        if result.returncode == 0:
            print(f"✅ Tests completed successfully ({execution_time:.1f}s)")
            return True
        else:
            print(f"❌ Tests failed ({execution_time:.1f}s)")
            if args.verbose and result.stdout:
                print("Test output:")
                print(result.stdout)
            return False

    def validate_examples(self, args: argparse.Namespace) -> bool:
        """Validate all examples."""
        print("🔍 Validating Example Structure")
        print("-" * 40)

        issues = []

        for example_name, example_info in self.examples.items():
            example_file = self.examples_dir / example_info['file']

            if not example_file.exists():
                issues.append(f"Missing example file: {example_file}")
                continue

            # Check if file is executable
            if not os.access(example_file, os.X_OK):
                issues.append(f"Example not executable: {example_file}")

            # Check for required imports
            try:
                with open(example_file, 'r') as f:
                    content = f.read()
                    if 'import evojump' not in content:
                        issues.append(f"Missing evojump import: {example_file}")
            except Exception as e:
                issues.append(f"Cannot read example file {example_file}: {e}")

        if issues:
            print("❌ Validation Issues Found:")
            for issue in issues:
                print(f"   • {issue}")
            return False

        print("✅ All examples validated successfully")
        return True

    def generate_summary_report(self, results: Dict[str, Tuple[bool, float]]) -> str:
        """Generate a summary report of all example runs."""
        successful = sum(1 for success, _ in results.values() if success)
        total = len(results)
        total_time = sum(exec_time for _, exec_time in results.values())

        report = f"""
📊 EvoJump Examples Validation Summary
{'=' * 50}

Results: {successful}/{total} examples passed
Total execution time: {total_time:.1f} seconds

Category Breakdown:
"""

        for category_name in self.categories.keys():
            category_results = {
                name: result for name, result in results.items()
                if self.examples[name]['category'] == category_name
            }

            if category_results:
                category_successful = sum(1 for success, _ in category_results.values() if success)
                category_total = len(category_results)
                report += f"  {self.categories[category_name]}: {category_successful}/{category_total}\n"

        report += "\nDetailed Results:\n"
        for example_name, (success, exec_time) in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            report += f"  {status}: {example_name} ({exec_time:.1f}s)\n"

        return report

    def run(self, args: argparse.Namespace) -> bool:
        """Main validation execution."""
        start_time = time.time()

        success = True

        # Validate example structure first
        if not self.validate_examples(args):
            success = False

        # Run examples if requested
        if not args.tests_only:
            example_results = self.run_all_examples(args)
            example_success = all(success for success, _ in example_results.values())

            if not example_success:
                success = False

        # Run tests if requested
        if not args.examples_only:
            test_success = self.run_tests(args)
            if not test_success:
                success = False

        # Generate final report
        end_time = time.time()
        total_time = end_time - start_time

        print("\n" + "=" * 60)
        print("🎉 EvoJump Validation Complete!")
        print("=" * 60)

        if success:
            print("✅ ALL VALIDATIONS PASSED!")
            print("🚀 EvoJump framework is working correctly!")
        else:
            print("❌ Some validations failed.")
            print("🔧 Please check the issues above.")

        print(f"⏱️  Total execution time: {total_time:.1f} seconds")

        if not args.tests_only and 'example_results' in locals():
            print("\n" + self.generate_summary_report(example_results))

        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive EvoJump Examples Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Run quick validation (skip slow examples)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Run tests in parallel where possible"
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure"
    )

    parser.add_argument(
        "--examples-only",
        action="store_true",
        help="Only run examples, skip testing"
    )

    parser.add_argument(
        "--tests-only",
        action="store_true",
        help="Only run tests, skip examples"
    )

    args = parser.parse_args()

    # Initialize and run validation
    validator = ExamplesValidator()
    success = validator.run(args)

    # Exit with appropriate code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
