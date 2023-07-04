# build the virtual environment and install all dependencies
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
cd py-polars && maturin develop --release -- -C codegen-units=16 -C lto=thin -C target-cpu=native

