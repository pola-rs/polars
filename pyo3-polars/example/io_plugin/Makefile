
SHELL=/bin/bash

.venv:  ## Set up virtual environment
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

install: .venv
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop -m io_plugin/Cargo.toml
	cd ../../../py-polars && maturin develop

clean:
	-@rm -r .venv
	-@cd extend_polars && cargo clean


run: install
	source .venv/bin/activate && python run.py
