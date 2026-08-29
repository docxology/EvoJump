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
