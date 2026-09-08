# EvoJump Examples

This directory contains comprehensive examples demonstrating the usage of the EvoJump package for evolutionary ontogenetic analysis.

## Example Categories

### 🚀 **Basic Usage Examples**
- **`working_demo.py`** - Working demonstration of all core features
- **`simple_orchestrator.py`** - Minimal end-to-end pipeline demo (writes `simple_orchestrator_outputs/`)

### 📊 **Advanced Analytics Examples**
- **`comprehensive_demo.py`** - Full analysis pipeline with all modules
- **`comprehensive_advanced_analytics_demo.py`** - Advanced statistical methods demonstration
- **`advanced_features_demo.py`** - Advanced stochastic process models

### 🎨 **Visualization Examples**
- **`animation_demo.py`** - Basic animation generation
- **`enhanced_animation_demo.py`** - Multi-condition and comparative animations
- **`comprehensive_animation_demo.py`** - Advanced animation with multiple types
- **`simple_animation_demo.py`** - Simple trajectory animation

### 🧬 **Case Study Examples**
- **`drosophila_case_study.py`** - Complete fruit fly biology analysis (see `../tests/test_drosophila_case_study.py`)

### ⚡ **Performance Examples**
- **`performance_benchmarks.py`** - Performance testing and benchmarking

> Superseded variants (`basic_usage_fixed.py`, `thin_orchestrator_examples.py`, `thin_orchestrator_working.py`) live in `archive/` and are not listed above.

## Running Examples

```bash
# Basic usage demonstration
python examples/working_demo.py

python examples/simple_orchestrator.py

# Comprehensive analysis
python examples/comprehensive_demo.py

# Advanced analytics
python examples/comprehensive_advanced_analytics_demo.py

# Animation examples
python examples/animation_demo.py
python examples/enhanced_animation_demo.py

# Performance benchmarking
python examples/performance_benchmarks.py

# Run all examples (for testing; from the repository root)
python run_all_examples.py
```

## Example Structure

Each example follows a consistent pattern:

1. **Data Generation** - Create synthetic developmental data
2. **Data Loading** - Load and validate data using DataCore
3. **Model Fitting** - Fit stochastic process models using JumpRope
4. **Analysis** - Perform cross-sectional and evolutionary analysis
5. **Visualization** - Generate plots and animations
6. **Reporting** - Save results and comprehensive reports

## Key Features Demonstrated

- ✅ **Data Management** - Loading, validation, preprocessing
- ✅ **Stochastic Modeling** - Jump-diffusion, OU, geometric, FBM, CIR, Levy processes
- ✅ **Cross-Sectional Analysis** - Distribution fitting, statistical comparisons
- ✅ **Advanced Analytics** - Bayesian, network, causal, dimensionality reduction
- ✅ **Evolutionary Analysis** - Population genetics, heritability, selection
- ✅ **Visualization** - Static plots, animations, interactive graphics
- ✅ **Performance** - Benchmarking and optimization

## Output Formats

Examples generate multiple output types:
- **PNG/JPG plots** - Static visualizations
- **GIF animations** - Dynamic developmental processes
- **JSON reports** - Comprehensive analysis results
- **CSV data** - Processed datasets and results


Output directories are CWD-relative (the repo root when run via `run_all_examples.py`): `demo_outputs/`, `evojump_outputs/`, `comprehensive_analytics_outputs/`, `comprehensive_animation_outputs/`, `enhanced_animations/`, `simple_animation_outputs/`, `simple_orchestrator_outputs/`, `animation_outputs/`, `outputs/figures/`, `outputs/benchmarks/`, and `drosophila_case_study_outputs/`.

## Scientific Applications

The examples demonstrate applications in:
- **Developmental Biology** - Ontogenetic trajectory analysis
- **Evolutionary Biology** - Population dynamics and selection
- **Quantitative Genetics** - Heritability and genetic correlations
- **Systems Biology** - Complex trait modeling
- **Agricultural Research** - Crop development optimization
- **Medical Research** - Disease progression modeling

