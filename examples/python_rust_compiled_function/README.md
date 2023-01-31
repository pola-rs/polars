# Compile Custom Rust functions and use in python polars

## Compile a development binary in your current environment

```sh
pip install -U maturin && maturin develop
```

## Run

```sh
python example.py
```

## Compile a **release** build

```sh
maturin develop --release
```
