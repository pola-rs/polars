Please only edit files in the `template*/` folders, then run the `template.py` script to regenerate
them. Do not edit the files in the generated `polars-runtime-*` folders directly.

To test builds locally, run the following commands from the root of the repo. The `polars` package
is a pure Python wheel, a wrapper around the chosen runtime; it is interpreter-agnostic
(free-threaded or not). Keep in mind that currently the `rt32` is always required by the `polars`
package.

The `abi3t` wheels target the stable free-threaded ABI introduced in Python 3.15 and work on both
GIL and free-threaded Python 3.15+. During build `maturin` selects the ABI automatically given the
provided interpreter.

```sh
$ uv venv --python 3.x  # or 3.xt for a free-threaded interpreter
$ source .venv/bin/activate
$ python -c "import sys; print(sys._is_gil_enabled())"
$ uv pip install maturin
$ uv build py-polars --out-dir dist
$ maturin build \
  --interpreter python3.x \
  --manifest-path py-polars/runtime/polars-runtime-xx/Cargo.toml \
  --out dist \
  --profile dev
$ uv pip install "polars[rtxx]" --find-links dist --no-index
$ python -c "import polars; polars.show_versions()"
```
