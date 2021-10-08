
changelog-python:
	docker run -it --env-file .env --rm -v "$(pwd)":/usr/local/src/your-app githubchangeloggenerator/github-changelog-generator -u pola-rs -p project --exclude-labels rust

changelog-rust:
	docker run -it --env-file .env --rm -v "$(pwd)":/usr/local/src/your-app githubchangeloggenerator/github-changelog-generator -u pola-rs -p project --exclude-labels python
