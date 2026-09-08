#!/bin/bash

# Build EvoJump Paper from Modular Markdown Files
# This script combines all section files into a complete paper and generates PDF

set -eo pipefail  # Exit on error; a failed pandoc/python in a pipeline must not be masked by tee/grep

echo "════════════════════════════════════════════════════════════════"
echo "Building EvoJump Paper"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check for pandoc
if ! command -v pandoc &> /dev/null; then
    echo "❌ Error: pandoc is not installed"
    echo "Install with: brew install pandoc (macOS) or apt-get install pandoc (Linux)"
    exit 1
fi

# Check for LaTeX
if ! command -v pdflatex &> /dev/null; then
    echo "⚠️  Warning: pdflatex not found. PDF generation may fail."
    echo "Install with: brew install --cask mactex (macOS)"
fi

# Set directories
PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTIONS_DIR="$PAPER_DIR/sections"
OUTPUT_DIR="$PAPER_DIR/output"
COMBINED_MD="$OUTPUT_DIR/combined_paper.md"

# Resolve the Python interpreter: prefer the project's uv-managed .venv
# (override by setting PYTHON=... when invoking this script).
PY="${PYTHON:-$PAPER_DIR/../.venv/bin/python}"
if [ ! -x "$PY" ]; then
    PY="$(command -v python3 || true)"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "📁 Paper directory: $PAPER_DIR"
echo "📄 Sections directory: $SECTIONS_DIR"
echo ""

# Create combined markdown file
echo "🔨 Combining markdown sections..."

# Author information for the PDF is provided by author_info.tex, so the
# combined markdown does not need YAML frontmatter for LaTeX generation.

# Clear/initialize the combined markdown file
> "$COMBINED_MD"

# Combine all section files in order
for i in 01 02 03 04 05 06 06a 07 08 09 10 11 12; do
    # Find files matching the pattern
    shopt -s nullglob
    for section_file in "$SECTIONS_DIR/${i}_"*.md; do
        if [ -f "$section_file" ]; then
            section_name=$(basename "$section_file" .md)
            echo "   ✓ Adding: $section_name"

            # Add to combined markdown (without YAML frontmatter)
            echo "" >> "$COMBINED_MD"
            echo "<!-- Section: $section_name -->" >> "$COMBINED_MD"
            echo "" >> "$COMBINED_MD"
            cat "$section_file" >> "$COMBINED_MD"
            echo "" >> "$COMBINED_MD"
            echo "" >> "$COMBINED_MD"
        fi
    done
done

echo ""
echo "📝 Combined markdown created: $(basename $COMBINED_MD)"
echo "   Size: $(wc -l < "$COMBINED_MD") lines"
echo ""

# Generate figures
echo ""
echo "🎨 Generating figures..."
if [ -f "$PAPER_DIR/render_figures.py" ]; then
    cd "$PAPER_DIR"
    "$PY" render_figures.py --figures_dir figures/ || {
        echo "❌ Figure generation failed. Build aborted so stale/missing figures are not shipped."
        exit 1
    }
    echo "✅ Main figures generated successfully!"
    echo "   Figures directory: $PAPER_DIR/figures/"
else
    echo "⚠️  Render script not found at $PAPER_DIR/render_figures.py"
    echo "   Figures will need to be generated manually."
fi

# Generate Drosophila case study figures
echo ""
echo "🧬 Generating Drosophila case study figures..."
if [ -f "$PAPER_DIR/render_drosophila_figures.py" ]; then
    cd "$PAPER_DIR"
    "$PY" render_drosophila_figures.py || {
        echo "❌ Drosophila figure generation failed. Build aborted so stale/missing figures are not shipped."
        exit 1
    }
    echo "✅ Drosophila figures generated successfully!"
else
    echo "⚠️  Drosophila render script not found at $PAPER_DIR/render_drosophila_figures.py"
    echo "   Drosophila figures will need to be generated manually."
fi

# Create PDF-ready markdown with proper YAML frontmatter
PDF_MD="$OUTPUT_DIR/pdf_ready.md"
cat > "$PDF_MD" << 'EOF'
---
title: "EvoJump: A Unified Framework for Stochastic Modeling of Evolutionary Ontogenetic Trajectories"
author: "Daniel Ari Friedman"
institute: "Active Inference Institute"
orcid: "0000-0001-6232-9096"
email: "daniel@activeinference.institute"
doi: "10.5281/zenodo.17229925"
date: "September 2025"
geometry: margin=1in
fontsize: 11pt
linestretch: 1.05
numbersections: true
toc: true
toc-depth: 3
link-citations: true
documentclass: article
header-includes: |
  \usepackage{amsmath}
  \usepackage{amssymb}
  \usepackage{booktabs}
  \usepackage{float}
  \usepackage{graphicx}
  \usepackage{hyperref}
  \graphicspath{{figures/}}
  \numberwithin{equation}{section}
---

EOF

# Add the content to the PDF-ready file
cat "$COMBINED_MD" >> "$PDF_MD"

# Generate PDF (remove any stale artifact first so a failed build cannot pass
# on a leftover file; with pipefail, pandoc's status is no longer masked by tee)
echo ""
echo "📄 Generating PDF..."
rm -f "$OUTPUT_DIR/evojump_paper.pdf"
pandoc "$PDF_MD" \
    -o "$OUTPUT_DIR/evojump_paper.pdf" \
    --pdf-engine=pdflatex \
    --number-sections \
    --toc \
    --toc-depth=3 \
    --include-in-header="$PAPER_DIR/author_info.tex" \
    --lua-filter="$PAPER_DIR/number-equations.lua" \
    2>&1 | tee "$OUTPUT_DIR/build.log" || true

if [ -f "$OUTPUT_DIR/evojump_paper.pdf" ]; then
    echo ""
    echo "✅ PDF successfully generated!"
    echo "📄 Output: $OUTPUT_DIR/evojump_paper.pdf"
    echo "   Size: $(du -h "$OUTPUT_DIR/evojump_paper.pdf" | cut -f1)"
else
    echo ""
    echo "❌ PDF generation failed. Check build.log for details."
    exit 1
fi

# Generate HTML version as well
echo ""
echo "🌐 Generating HTML version..."
rm -f "$OUTPUT_DIR/evojump_paper.html"
pandoc "$COMBINED_MD" \
    -o "$OUTPUT_DIR/evojump_paper.html" \
    --standalone \
    --toc \
    --toc-depth=3 \
    --number-sections \
    --katex \
    --css=https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css \
    2>&1 | grep -v "WARNING" || true

if [ -f "$OUTPUT_DIR/evojump_paper.html" ]; then
    echo "✅ HTML successfully generated!"
    echo "🌐 Output: $OUTPUT_DIR/evojump_paper.html"
else
    echo "❌ HTML generation failed."
    exit 1
fi

# Generate Word version
echo ""
echo "📝 Generating Word (.docx) version..."
rm -f "$OUTPUT_DIR/evojump_paper.docx"
if [ -f "$PAPER_DIR/reference.docx" ]; then
    pandoc "$COMBINED_MD" \
        -o "$OUTPUT_DIR/evojump_paper.docx" \
        --reference-doc="$PAPER_DIR/reference.docx" \
        --toc \
        --number-sections \
        2>&1 | grep -v "WARNING" || true
else
    pandoc "$COMBINED_MD" \
        -o "$OUTPUT_DIR/evojump_paper.docx" \
        --toc \
        --number-sections \
        2>&1 | grep -v "WARNING" || true
fi

if [ -f "$OUTPUT_DIR/evojump_paper.docx" ]; then
    echo "✅ Word document successfully generated!"
    echo "📝 Output: $OUTPUT_DIR/evojump_paper.docx"
else
    echo "❌ Word generation failed."
    exit 1
fi


echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✨ Build Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📁 Output directory: $OUTPUT_DIR"
echo ""
echo "📄 Generated files:"
ls -lh "$OUTPUT_DIR"/evojump_paper.* 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
echo ""
echo "🚀 To view the paper:"
echo "   PDF:  open $OUTPUT_DIR/evojump_paper.pdf"
echo "   HTML: open $OUTPUT_DIR/evojump_paper.html"
echo ""
echo "📊 Build artifacts:"
echo "   Combined source: $OUTPUT_DIR/combined_paper.md"
echo "   Build log: $OUTPUT_DIR/build.log"
echo ""
