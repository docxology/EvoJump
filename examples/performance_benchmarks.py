"""
Performance Benchmarking for EvoJump

This script benchmarks the performance of various EvoJump components
to identify bottlenecks and optimization opportunities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import evojump as ej


def benchmark_function(func, *args, **kwargs):
    """Benchmark a function and return execution time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result


def benchmark_data_loading(n_samples_list):
    """Benchmark data loading performance."""
    print("\n=== BENCHMARKING DATA LOADING ===")
    results = {}
    
    for n_samples in n_samples_list:
        print(f"Testing with {n_samples} samples...")
        
        # Create synthetic data
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5] * n_samples,
            'phenotype1': np.random.normal(10, 2, 5 * n_samples),
            'phenotype2': np.random.normal(20, 3, 5 * n_samples)
        })
        
        # Benchmark CSV loading
        csv_str = data.to_csv()
        time_taken, _ = benchmark_function(
            ej.DataCore.load_from_csv,
            pd.io.common.StringIO(csv_str),
            time_column='time'
        )
        
        results[n_samples] = time_taken
        print(f"   Time: {time_taken:.4f} seconds")
    
    return results


def benchmark_model_fitting(n_samples_list, model_types):
    """Benchmark model fitting performance."""
    print("\n=== BENCHMARKING MODEL FITTING ===")
    results = {}
    
    for model_type in model_types:
        print(f"\nTesting {model_type} model...")
        results[model_type] = {}
        
        for n_samples in n_samples_list:
            # Create test data
            data = pd.DataFrame({
                'time': [1, 2, 3, 4, 5] * n_samples,
                'phenotype1': np.random.normal(10, 2, 5 * n_samples)
            })
            
            data_core = ej.DataCore.load_from_csv(
                pd.io.common.StringIO(data.to_csv()),
                time_column='time'
            )
            
            # Benchmark model fitting
            time_taken, _ = benchmark_function(
                ej.JumpRope.fit,
                data_core,
                model_type=model_type
            )
            
            results[model_type][n_samples] = time_taken
            print(f"   {n_samples} samples: {time_taken:.4f} seconds")
    
    return results


def benchmark_trajectory_generation(n_trajectories_list, n_timepoints_list):
    """Benchmark trajectory generation performance."""
    print("\n=== BENCHMARKING TRAJECTORY GENERATION ===")
    results = {}
    
    # Create a simple model
    data = pd.DataFrame({
        'time': [1, 2, 3, 4, 5] * 10,
        'phenotype1': np.random.normal(10, 2, 50)
    })
    
    data_core = ej.DataCore.load_from_csv(
        pd.io.common.StringIO(data.to_csv()),
        time_column='time'
    )
    
    for n_trajectories in n_trajectories_list:
        print(f"\nTesting with {n_trajectories} trajectories...")
        results[n_trajectories] = {}
        
        for n_timepoints in n_timepoints_list:
            model = ej.JumpRope.fit(data_core, model_type='jump-diffusion')
            model.time_points = np.linspace(0, 10, n_timepoints)
            
            time_taken, _ = benchmark_function(
                model.generate_trajectories,
                n_samples=n_trajectories,
                x0=10.0
            )
            
            results[n_trajectories][n_timepoints] = time_taken
            print(f"   {n_timepoints} timepoints: {time_taken:.4f} seconds")
    
    return results


def benchmark_visualization(n_trajectories_list):
    """Benchmark visualization performance."""
    print("\n=== BENCHMARKING VISUALIZATION ===")
    results = {}
    
    visualizer = ej.TrajectoryVisualizer()
    
    for n_trajectories in n_trajectories_list:
        print(f"\nTesting with {n_trajectories} trajectories...")
        
        # Create model with trajectories
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5] * 10,
            'phenotype1': np.random.normal(10, 2, 50)
        })
        
        data_core = ej.DataCore.load_from_csv(
            pd.io.common.StringIO(data.to_csv()),
            time_column='time'
        )
        
        model = ej.JumpRope.fit(data_core, model_type='jump-diffusion')
        model.generate_trajectories(n_samples=n_trajectories, x0=10.0)
        
        # Benchmark different visualizations
        viz_types = {
            'trajectories': lambda: visualizer.plot_trajectories(model, interactive=False),
            'heatmap': lambda: visualizer.plot_heatmap(model, time_resolution=30, phenotype_resolution=30),
            'violin': lambda: visualizer.plot_violin(model),
            'ridge': lambda: visualizer.plot_ridge(model, n_distributions=5),
            'phase_portrait': lambda: visualizer.plot_phase_portrait(model)
        }
        
        results[n_trajectories] = {}
        for viz_name, viz_func in viz_types.items():
            time_taken, fig = benchmark_function(viz_func)
            plt.close(fig)
            results[n_trajectories][viz_name] = time_taken
            print(f"   {viz_name}: {time_taken:.4f} seconds")
    
    return results


def benchmark_analytics(n_samples_list):
    """Benchmark analytics performance."""
    print("\n=== BENCHMARKING ANALYTICS ===")
    results = {}
    
    for n_samples in n_samples_list:
        print(f"\nTesting with {n_samples} samples...")
        
        # Create test data
        data = pd.DataFrame({
            'time': np.arange(1, n_samples + 1),
            'phenotype1': np.random.normal(10, 2, n_samples),
            'phenotype2': np.random.normal(20, 3, n_samples)
        })
        
        analytics = ej.AnalyticsEngine(data, time_column='time')
        
        # Benchmark different analyses
        analysis_types = {
            'copula': lambda: analytics.copula_analysis('phenotype1', 'phenotype2'),
            'extreme_value': lambda: analytics.extreme_value_analysis('phenotype1'),
            'regime_switching': lambda: analytics.regime_switching_analysis('phenotype1', n_regimes=3),
        }
        
        results[n_samples] = {}
        for analysis_name, analysis_func in analysis_types.items():
            try:
                time_taken, _ = benchmark_function(analysis_func)
                results[n_samples][analysis_name] = time_taken
                print(f"   {analysis_name}: {time_taken:.4f} seconds")
            except Exception as e:
                print(f"   {analysis_name}: FAILED ({str(e)})")
                results[n_samples][analysis_name] = None
    
    return results


def plot_benchmark_results(results_dict, output_dir):
    """Plot benchmark results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot data loading benchmarks
    if 'data_loading' in results_dict:
        fig, ax = plt.subplots(figsize=(10, 6))
        samples = list(results_dict['data_loading'].keys())
        times = list(results_dict['data_loading'].values())
        ax.plot(samples, times, marker='o', linewidth=2)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Data Loading Performance')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'benchmark_data_loading.png', dpi=300)
        plt.close()
    
    # Plot model fitting benchmarks
    if 'model_fitting' in results_dict:
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_type, data in results_dict['model_fitting'].items():
            samples = list(data.keys())
            times = list(data.values())
            ax.plot(samples, times, marker='o', linewidth=2, label=model_type)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Model Fitting Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'benchmark_model_fitting.png', dpi=300)
        plt.close()
    
    # Plot visualization benchmarks
    if 'visualization' in results_dict:
        fig, ax = plt.subplots(figsize=(12, 6))
        viz_types = set()
        for n_traj, data in results_dict['visualization'].items():
            viz_types.update(data.keys())
        
        x_pos = np.arange(len(results_dict['visualization']))
        width = 0.15
        
        for i, viz_type in enumerate(sorted(viz_types)):
            times = [data.get(viz_type, 0) for data in results_dict['visualization'].values()]
            ax.bar(x_pos + i * width, times, width, label=viz_type)
        
        ax.set_xlabel('Number of Trajectories')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Visualization Performance')
        ax.set_xticks(x_pos + width * 2)
        ax.set_xticklabels(list(results_dict['visualization'].keys()))
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / 'benchmark_visualization.png', dpi=300)
        plt.close()


def generate_performance_report(all_results, output_dir):
    """Generate a comprehensive performance report."""
    output_dir = Path(output_dir)
    report_path = output_dir / 'performance_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EVOJUMP PERFORMANCE BENCHMARK REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # Data Loading
        if 'data_loading' in all_results:
            f.write("DATA LOADING PERFORMANCE:\n")
            f.write("-" * 50 + "\n")
            for n_samples, time_taken in all_results['data_loading'].items():
                f.write(f"  {n_samples:6d} samples: {time_taken:8.4f} seconds\n")
            f.write("\n")
        
        # Model Fitting
        if 'model_fitting' in all_results:
            f.write("MODEL FITTING PERFORMANCE:\n")
            f.write("-" * 50 + "\n")
            for model_type, data in all_results['model_fitting'].items():
                f.write(f"  {model_type}:\n")
                for n_samples, time_taken in data.items():
                    f.write(f"    {n_samples:6d} samples: {time_taken:8.4f} seconds\n")
            f.write("\n")
        
        # Trajectory Generation
        if 'trajectory_generation' in all_results:
            f.write("TRAJECTORY GENERATION PERFORMANCE:\n")
            f.write("-" * 50 + "\n")
            for n_traj, data in all_results['trajectory_generation'].items():
                f.write(f"  {n_traj} trajectories:\n")
                for n_time, time_taken in data.items():
                    f.write(f"    {n_time:6d} timepoints: {time_taken:8.4f} seconds\n")
            f.write("\n")
        
        # Visualization
        if 'visualization' in all_results:
            f.write("VISUALIZATION PERFORMANCE:\n")
            f.write("-" * 50 + "\n")
            for n_traj, data in all_results['visualization'].items():
                f.write(f"  {n_traj} trajectories:\n")
                for viz_type, time_taken in data.items():
                    f.write(f"    {viz_type:20s}: {time_taken:8.4f} seconds\n")
            f.write("\n")
        
        # Analytics
        if 'analytics' in all_results:
            f.write("ANALYTICS PERFORMANCE:\n")
            f.write("-" * 50 + "\n")
            for n_samples, data in all_results['analytics'].items():
                f.write(f"  {n_samples} samples:\n")
                for analysis_type, time_taken in data.items():
                    if time_taken is not None:
                        f.write(f"    {analysis_type:20s}: {time_taken:8.4f} seconds\n")
                    else:
                        f.write(f"    {analysis_type:20s}: FAILED\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"\n✅ Performance report saved to: {report_path}")


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("EVOJUMP PERFORMANCE BENCHMARKS")
    print("=" * 70)
    
    np.random.seed(42)
    output_dir = Path("outputs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Benchmark data loading
    all_results['data_loading'] = benchmark_data_loading([10, 50, 100, 500])
    
    # Benchmark model fitting
    all_results['model_fitting'] = benchmark_model_fitting(
        [10, 50, 100],
        ['jump-diffusion', 'fractional-brownian', 'cir', 'levy']
    )
    
    # Benchmark trajectory generation
    all_results['trajectory_generation'] = benchmark_trajectory_generation(
        [50, 100, 200],
        [50, 100, 200]
    )
    
    # Benchmark visualization
    all_results['visualization'] = benchmark_visualization([50, 100, 200])
    
    # Benchmark analytics
    all_results['analytics'] = benchmark_analytics([100, 500, 1000])
    
    # Plot results
    print("\n=== GENERATING PLOTS ===")
    plot_benchmark_results(all_results, output_dir)
    
    # Generate report
    print("\n=== GENERATING REPORT ===")
    generate_performance_report(all_results, output_dir)
    
    print("\n" + "=" * 70)
    print("✅ BENCHMARKING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Results saved to: {output_dir.absolute()}")
    print("\n📊 Performance Summary:")
    print(f"   • Data loading tested with up to 500 samples")
    print(f"   • Model fitting tested with 4 model types")
    print(f"   • Trajectory generation tested with up to 200 trajectories")
    print(f"   • 5 visualization types benchmarked")
    print(f"   • 3 analytics methods tested")
    print("\n🚀 Check the report for detailed timings!")


if __name__ == '__main__':
    main()
