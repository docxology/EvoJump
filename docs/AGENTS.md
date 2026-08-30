# Agent notes — EvoJump/docs

- This is a Sphinx `.rst` tree; navigation is driven by the toctrees in
  `index.rst`. When adding a page, add it to a toctree or it is orphaned.
- Keep page titles and cross-references in RST conventions (`:ref:`,
  `:doc:`); the `conf.py` at this level is the single Sphinx config.
- Root `README.md` is the human entry point; `paper/` holds the
  publication-track paper source (modular sections + `build_paper.sh`),
  documented in `docs/manuscript/MANUSCRIPT_STATUS.md`.
- Factual claims added here must trace to `src/`, `tests/`, `pyproject.toml`,
  or existing docs. If unverifiable, write "Not documented in repo — needs
  owner input" instead of inventing content.
- Model-type and method claims: `JumpRope.fit` accepts seven model types
  (`jump-diffusion` default, `ornstein-uhlenbeck`, `geometric-jump-diffusion`,
  `compound-poisson`, `fractional-brownian`, `cir`, `levy`) — see
  `src/evojump/jumprope.py` `fit()`. Verify method existence in
  `src/evojump/analytics_engine.py` (e.g. `shortest_path_analysis`,
  BOCPD change-point detection) before documenting.
- Test invocation under load: `.venv/bin/python -m pytest tests/` — do not
  route through `uv run` (stalls under heavy machine load; observed during
  the 2026-08-30 audit pass).
- Known stale-spot: `src/evojump/jumprope.py` `fit()` docstring lists only
  three model types while the implementation accepts seven — trust the code.
