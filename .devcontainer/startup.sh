rustc --version
cargo --version
rustup update stable

rustc --version
cargo --version

cd py-polars

# construct a virtual environment as required by maturin
python -m venv .venv
# source .venv/bin/activate

# upgrade pip and install dev-dependencies
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-dev.txt

.venv/bin/maturin develop --release -- -C target-cpu=native
.venv/bin/pytest -n auto --dist worksteal
