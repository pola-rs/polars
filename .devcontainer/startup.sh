cd py-polars

# construct a virtual environment as required by maturin
python -m venv .venv
source .venv/bin/activate

# upgrade pip and install dev-dependencies
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt

.venv/bin/maturin develop --release -- -C target-cpu=native
.venv/bin/pytest -n auto --dist worksteal
