# Manuscript status — EvoJump

**Repo type:** Publication-track research project, with the paper source
already present but outside the standard `manuscript/` layout.

**Evidence checked:** `paper/` contains modular section files
(`paper/sections/01_abstract.md` … `12_code.md`, matching the section map in
`paper/README.md`), `paper/paper.md` (assembled manuscript with YAML
metadata), `paper/build_paper.sh`, `paper/latex_template.tex`,
`paper/figures/`, and `paper/paper_verification_report.md` (build report:
85-page PDF, marked SUCCESS). The root ships research outputs
(`drosophila_case_study_outputs/`, `simple_animation_outputs/`,
`sample_developmental_data.csv`) and a research README.

**Why no `manuscript/` was created in this audit:** the publication-track
content already exists in full under `paper/`; creating a parallel
`manuscript/` stub tree would duplicate it and risk divergence. The
docxology/template standard layout (`manuscript/` at repo top level with
`config.yaml`, section files, `references.bib`) is NOT satisfied by this
repo — that is a structural deviation, recorded here rather than papered over.

**What would trigger standardizing:** migrating `paper/sections/` into
`manuscript/NN_*.md` with a `manuscript/config.yaml` (title/author metadata
from `paper/paper.md` — note `paper/README.md` credits Daniel Ari Friedman
while `paper/paper.md` lists "EvoJump Development Team"; an owner should
reconcile that discrepancy) and `references.bib`. Needs owner input before
any migration; no content was rewritten in this audit.
