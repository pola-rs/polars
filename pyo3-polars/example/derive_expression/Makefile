
SHELL=/bin/bash
POLARS_REPO=../../..

.venv:  ## Set up virtual environment
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

# The plugin is built against the Polars crates of this repository.
.PHONY: polars
polars: .venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && \
	uv pip install --no-deps -e $(POLARS_REPO)/py-polars && \
	maturin develop -m $(POLARS_REPO)/py-polars/runtime/polars-runtime-32/Cargo.toml --features backtrace_filter --uv

install: polars
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop -m expression_lib/Cargo.toml --uv

install-release: polars
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop --release -m expression_lib/Cargo.toml --uv

clean:
	-@rm -r .venv
	-@cd expression_lib && cargo clean


run: install
	source .venv/bin/activate && python run.py

run-release: install-release
	source .venv/bin/activate && python run.py

test: install
	source .venv/bin/activate && pytest tests
