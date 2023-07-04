# install rust
#curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
#source "$HOME/.cargo/env"

cd py-polars

# construct a virtual environment as required by maturin
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-lint.txt
python -m pip install -r docs/requirements-docs.txt

pip install maturin

maturin develop --release

pytest -n auto --dist worksteal
