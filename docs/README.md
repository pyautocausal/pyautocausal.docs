# PyAutoCausal Documentation

This directory contains the Sphinx documentation for PyAutoCausal using the Read the Docs theme.

## Building the Documentation

### Prerequisites

Make sure you have the documentation dependencies installed:

```bash
poetry install --with docs
```

### Building HTML Documentation

To build the HTML documentation:

```bash
cd docs
poetry run sphinx-build -b html . _build/html
```

Or using the Makefile:

```bash
cd docs
poetry run make html
```

The generated documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser to view it.

### Building on Read the Docs

The documentation is configured to build automatically on Read the Docs. The configuration is in `.readthedocs.yaml` at the project root.

## Documentation Structure

- `index.rst` - Main documentation entry point
- `getting-started.md` - Getting started guide
- `data-requirements.md` - Data requirements documentation
- `causal-methods.md` - Causal methods reference
- `design-document.md` - Design document
- `did_flow.md` - DiD decision flow guide
- `wrapper_functions.md` - Wrapper functions documentation
- `api/` - API reference documentation
- `conf.py` - Sphinx configuration file

## Adding New Pages

1. Create a new `.md` or `.rst` file in the `docs/` directory
2. Add the file to the toctree in `index.rst`
3. Rebuild the documentation

## Theme

This documentation uses the [Read the Docs Sphinx Theme](https://sphinx-rtd-theme.readthedocs.io/).

