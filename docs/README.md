# EvoJump documentation (Sphinx)

Sphinx sources for the EvoJump documentation site (also published on
Read the Docs per the root `README.md` badge). EvoJump is a framework for
evolutionary ontogenetic analysis treating development as a "cross-sectional
laser" sweep of phenotypic distributions — see root `README.md`.

## Files

| File | Contents |
|---|---|
| `index.rst` | Landing page and toctree |
| `installation.rst`, `quickstart.rst` | Getting started |
| `examples.rst`, `advanced_usage.rst`, `advanced_methods.rst` | Usage guides |
| `api_reference.rst` | API reference |
| `architecture.rst` | System architecture |
| `troubleshooting.rst` | Troubleshooting |
| `contributing.rst`, `changelog.rst` | Contribution and change notes |
| `conf.py` | Sphinx configuration |

## Building

The root `pyproject.toml` declares the package and its dev dependencies
(including pytest). Build the Sphinx docs with `sphinx-build` against this
directory (conf.py is the config); test the package with `pytest tests/`.
The publication-track paper source lives separately in `../paper/` —
see `manuscript/MANUSCRIPT_STATUS.md` in this folder.
