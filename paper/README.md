# EvoJump Paper

This directory contains the modular markdown source files for the EvoJump academic paper, which comprehensively describes the analytical basis of the EvoJump package.

## Author

**Daniel Ari Friedman**  
ORCID: [0000-0001-6232-9096](https://orcid.org/0000-0001-6232-9096)  
Email: daniel@activeinference.institute  
Affiliation: Active Inference Institute

## Paper Structure

The paper is organized into 12 modular sections:

1. **Abstract** (`01_abstract.md`) - Summary of the framework and contributions
2. **Introduction** (`02_introduction.md`) - Background, motivation, and conceptual framework
3. **Mathematical Foundations** (`03_mathematical_foundations.md`) - Rigorous mathematical formulation of all stochastic process models
4. **Statistical Methods** (`04_statistical_methods.md`) - Advanced analytical methods (wavelets, copulas, EVT, etc.)
5. **Implementation** (`05_implementation.md`) - Software architecture, algorithms, and computational details
6. **Results** (`06_results.md`) - Validation, benchmarking, and applications
7. **Discussion** (`07_discussion.md`) - Interpretation, limitations, and future directions
8. **Conclusion** (`08_conclusion.md`) - Summary and broader impact
9. **References** (`09_references.md`) - Complete bibliography
10. **Figures** (`10_figures.md`) - Figure reproducibility and technical details
11. **Glossary** (`11_glossary.md`) - Comprehensive symbol definitions (150+ entries)
12. **Code Listings** (`12_code.md`) - Complete implementation code and examples

## Building the Paper

### Prerequisites

- **pandoc**: Universal document converter
- **LaTeX** (for PDF generation): MacTeX (macOS) or TeX Live (Linux/Windows)
- **Optional**: pandoc-crossref for cross-references

Install on macOS:
```bash
brew install pandoc
brew install --cask mactex
```

### Build Commands

**Build all formats** (PDF, HTML, DOCX):
```bash
./build_paper.sh
```

**Build PDF only**:
```bash
pandoc combined_paper.md -o evojump_paper.pdf --pdf-engine=pdflatex
```

**Build HTML only**:
```bash
pandoc combined_paper.md -o evojump_paper.html --standalone --katex
```

## Output Files

All build outputs are automatically placed in the `output/` subdirectory:

- `output/evojump_paper.pdf` - Publication-ready PDF version (85 pages, 4.7 MB with figures)
- `output/evojump_paper.html` - Web-viewable HTML version with KaTeX math rendering
- `output/evojump_paper.docx` - Microsoft Word version for collaborative editing
- `output/combined_paper.md` - All sections combined into single markdown file (2,722 lines)
- `output/build.log` - Build process log for debugging
- `output/FINAL_COMPREHENSIVE_STATUS.md` - Complete build verification and status

## Paper Content

### Mathematical Content

The paper includes:
- Formal mathematical definitions of all stochastic processes
- Rigorous derivations of parameter estimation procedures
- Analytical properties (stationarity, autocorrelation, etc.)
- Convergence theorems and asymptotic results

### Statistical Methods

Comprehensive coverage of:
- Continuous wavelet transform theory and implementation
- Copula families (Gaussian, Clayton, Frank) and dependence measures
- Peaks-over-threshold and block maxima methods for extreme values
- Regime-switching detection via K-means clustering
- Information-theoretic measures (entropy, mutual information, transfer entropy)

### Computational Implementation

Detailed discussion of:
- Software architecture and design principles
- Algorithmic implementations with pseudocode
- Performance optimization strategies (vectorization, JIT, parallelization)
- Testing framework and validation procedures
- Package management with UV

### Validation and Results

Extensive validation including:
- Synthetic data with known properties
- Parameter recovery studies
- Comparison with analytical solutions
- Performance benchmarking
- Real biological data applications

## Auto-Numbering Features

The paper includes comprehensive auto-numbering:

### Sections
- Automatically numbered via pandoc's `--number-sections` flag
- Hierarchical: 1, 1.1, 1.1.1, etc.
- Included in table of contents with page numbers

### Equations
- Label equations with `\label{eq:unique_name}`
- Reference with `\eqref{eq:unique_name}` (provided by amsmath)
- Example: `$$dX_t = \mu dt + \sigma dW_t \label{eq:sde}$$`

### Figures
- Use custom `\figref{name}` command
- Label figures with `\label{fig:unique_name}`
- Figures stored in `figures/` directory

## Editing Guidelines

When editing section files:

1. **Mathematical notation**: Use LaTeX syntax within `$...$` (inline) or `$$...$$` (display)
2. **Equation labels**: Add `\label{eq:name}` to all important equations
3. **Figure references**: Use `\figref{name}` for cross-references
4. **Citations**: Use Author (Year) format, e.g., `(Smith 2020)`
5. **Code blocks**: Use triple backticks with language specification
6. **Tables**: Use markdown table syntax
7. **Sectioning**: Use markdown headers `#`, `##`, `###` for hierarchical structure

## Dependencies

The paper documents these key dependencies:

- **NumPy** ≥ 1.21.0: Array operations
- **SciPy** ≥ 1.7.0: Scientific computing
- **pandas** ≥ 1.3.0: Data structures
- **matplotlib** ≥ 3.5.0: Static plotting
- **Plotly** ≥ 5.0.0: Interactive visualization
- **scikit-learn** ≥ 1.0.0: Machine learning
- **PyWavelets** ≥ 1.3.0: Wavelet analysis
- **NetworkX** ≥ 2.6.0: Network analysis
- **StatsModels** ≥ 0.13.0: Statistical models
- **Seaborn** ≥ 0.11.0: Enhanced visualization

## Contribution

To contribute to the paper:

1. Edit appropriate section file in `sections/`
2. Rebuild using `./build_paper.sh`
3. Review generated PDF
4. Submit changes via pull request

## Citation

If you use EvoJump in your research, please cite:

```
Friedman, D. A. (2025). EvoJump: A Unified Framework for Stochastic 
Modeling of Evolutionary Ontogenetic Trajectories. 
[Journal details to be added upon publication]
```

## License

The paper text is licensed under CC BY 4.0. The EvoJump software is licensed under Apache License 2.0.

## Contact

**Daniel Ari Friedman**  
Email: daniel@activeinference.institute  
ORCID: [0000-0001-6232-9096](https://orcid.org/0000-0001-6232-9096)

For questions about the paper:
- Create an issue at the GitHub repository
- Email the author directly

---

**Status**: Complete - ready for submission

**Word Count**: ~18,000 words across all sections (including code listings)

**Pages**: 85 (with figures and complete implementation code)

**Last Updated**: September 2025
