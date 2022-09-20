
changelog-python:
	docker run -it --env-file .env --rm -v "$(pwd)":/usr/local/src/your-app githubchangeloggenerator/github-changelog-generator -u pola-rs -p polars --exclude-labels rust

changelog-rust:
	docker run -it --env-file .env --rm -v "$(pwd)":/usr/local/src/your-app githubchangeloggenerator/github-changelog-generator -u pola-rs -p polars --exclude-labels python

fmt_toml:
	dprint fmt
