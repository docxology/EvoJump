Installation guide
==================

EvoJump supports Python 3.11 through 3.14. GitHub CI exercises Python 3.11
and 3.12 on Ubuntu. The source distribution declares the package, runtime
dependencies, and optional extras in its project metadata.

Install the checkout
--------------------

Create an environment and install from the repository.

.. code-block:: sh

   git clone https://github.com/docxology/EvoJump.git
   cd EvoJump
   uv venv --python 3.13
   uv pip install -e .

The checked-in lock targets macOS. On macOS, use ``uv sync --locked`` to
reproduce that environment. The editable-install recipe uses the package
metadata and follows the installation route used by Ubuntu CI.

Optional dependencies
---------------------

Install additional capabilities into the same environment as needed.

.. code-block:: sh

   uv pip install -e '.[test]'
   uv pip install -e '.[dev]'
   uv pip install -e '.[web]'

The ``gpu`` extra declares CuPy for Linux x86-64 with Python below 3.14 and
requires a compatible local CUDA runtime. The ``r-integration`` extra
requires an installed R runtime.

Verify the installation
-----------------------

Run these commands from the repository root.

.. code-block:: sh

   .venv/bin/python -c 'import evojump; print(evojump.__version__)'
   .venv/bin/python -m pytest tests/test_jumprope.py tests/test_laserplane.py -q

The manuscript recipe verifies a deterministic recurrence and an
independent cross-sectional statistic. See the
`manuscript directory <https://github.com/docxology/EvoJump/tree/main/docs/manuscript>`_.

Build the documentation
-----------------------

.. code-block:: sh

   .venv/bin/python -m sphinx -b html docs docs/_build/html
