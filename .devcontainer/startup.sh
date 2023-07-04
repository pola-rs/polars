cd py-polars

# construct a virtual environment as required by maturin
python -m venv .venv
source .venv/bin/activate

# upgrade pip and install dev-dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

@unset CONDA_PREFIX && maturin develop --release -- -C target-cpu=native
pytest -n auto --dist worksteal
