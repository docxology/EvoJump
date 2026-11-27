# EvoJump manuscripts

The current methods article checks the stochastic solver and cross-sectional
statistics against a deterministic recurrence and a seeded numerical fixture.
The complete authored study, its figures, and the native renderer are retained
in [the native edition](../publications/evojump-0.5.2/README.md). [Published study](https://doi.org/10.5281/zenodo.22664675)

## Render the current methods article

Install Git, uv, Pandoc with matching pandoc-crossref, and XeLaTeX.
From the EvoJump repository root, register a clean worktree in the pinned
Template checkout.

```sh
template_dir=/absolute/path/to/template
project_dir="$template_dir/projects/working/EvoJump"
git -C "$template_dir" switch --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
mkdir -p "$template_dir/projects/working"
git worktree add --detach "$project_dir" HEAD
(
  cd "$project_dir"
  uv venv --python 3.13
  uv pip install -e .
  MPLBACKEND=Agg .venv/bin/python -m docs.reproduce_trajectories
)
cd "$template_dir"
uv sync --locked --python 3.14 --no-dev
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/EvoJump
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/EvoJump
```

The PDF, HTML, native numerical receipt, plot, and validation reports are
under `projects/working/EvoJump/output/`. [Template prerequisites](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/docs/RUN_GUIDE.md)
