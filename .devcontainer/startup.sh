cd py-polars

# construct a virtual environment as required by maturin
python -m venv .venv
source .venv/bin/activate

# upgrade pip and install dev-dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

# Not required at this early stage...
# python -m pip install -r requirements-lint.txt
# python -m pip install -r docs/requirements-docs.txt

# maturin is a dependency declared in requirements-dev.txt
# pip install maturin

maturin develop --release

pytest -n auto --dist worksteal
