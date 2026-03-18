# Installation

Skyweaver targets Python **3.13+** on Linux, macOS, untested on windows, but should support WSL.  
We recommend installing inside a **virtual environment** so your setup stays isolated from system Python.

---

## 1. Virtual environment

You can use either `venv` (built into Python) **or** [uv](https://github.com/astral-sh/uv) (a modern fast manager).

### A) `venv` (built-in)

```bash
# Create and activate a venv in .venv/
python -m venv .venv
source .venv/bin/activate
```

### B) – uv (fast & modern)

```bash
# If you don’t have uv yet:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a venv in .venv/
uv venv
source .venv/bin/activate
```

---

## 2. Install Skyweaver

### A) From PyPI

```bash
pip install skyweaver
# or with uv:
uv pip install skyweaver
```

### B) From source

```bash
git clone https://github.com/amanchokshi/skyweaver.git
cd skyweaver

# Regular install
pip install .

# Editable install (recommended for development)
pip install -e .
```

Using uv:

```bash
uv pip install .
uv pip install -e .
```

---

## 3. Development install

If you want to run the test suite, build the docs, or contribute code, install with the dev extra defined in `pyproject.toml`:

```bash
pip install -e ".[dev]"
# or with uv:
uv pip install -e ".[dev]"
```

This installs additional dependencies for:

- **Testing** (`pytest`, `pytest-cov`, etc.)
- **Docs** (`mkdocs`, `mkdocs-material`, `mkdocstrings`, etc.)
- **Linting & formatting** (`ruff`, `black`, etc.)

---

You’re now ready to start simulating imaginary satellites with `Skyweaver` ✨
