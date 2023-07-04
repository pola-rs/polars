# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# install maturin
pip install maturin
cd py-polars && maturin develop --release -- -C codegen-units=16 -C lto=thin -C target-cpu=native
make test
