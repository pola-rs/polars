
changelog-python:
	docker run -it --env-file .env --rm -v "$(pwd)":/usr/local/src/your-app githubchangeloggenerator/github-changelog-generator -u pola-rs -p polars --exclude-labels rust

changelog-rust:
	docker run -it --env-file .env --rm -v "$(pwd)":/usr/local/src/your-app githubchangeloggenerator/github-changelog-generator -u pola-rs -p polars --exclude-labels python

fmt_toml:
	dprint fmt

coverage:
	@bash -c "\
		rustup override set nightly-2022-08-22; \
		source <(cargo llvm-cov show-env --export-prefix); \
		export CARGO_TARGET_DIR=\$$CARGO_LLVM_COV_TARGET_DIR; \
		export CARGO_INCREMENTAL=1; \
		cargo llvm-cov clean --workspace; \
		$(MAKE) -C py-polars venv; \
		source py-polars/venv/bin/activate; \
		$(MAKE) -C polars test; \
		$(MAKE) -C py-polars test-with-cov; \
		cargo llvm-cov --no-run --lcov --output-path coverage.lcov; \
		"
