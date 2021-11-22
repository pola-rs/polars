# Contributing to Polars

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Adding/Proposing new features

## We Develop with GitHub

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Want to discuss something?

I can imagine that some questions don't fit an issue.
Therefore there is also a [chat on Gitter](https://gitter.im/polars-rs/community).

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same
[MIT License](https://choosealicense.com/licenses/mit/) that covers the project.
Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issues](https://github.com/pola-rs/polars/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/pola-rs/polars/issues/new/choose).
**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Code formatting

We test the code formatting in the CI pipelines. If you don't want these to fail, you need to format:

- **Rust** code with `$ cargo fmt`
- **Python** code with [black](https://github.com/psf/black) (version 21.6b0) and [isort](https://github.com/PyCQA/isort) (version 5.9.2). Run both from the `py-polars` directory with `$ black . && isort .`

## Linting

We use linters to enforce code quality. This will be checked in CI.

- **Rust** We use [clippy](https://github.com/rust-lang/rust-clippy) as linter.
- **Python** We use [flake8](https://flake8.pycqa.org/en/latest/) as linter.

## Type checking

For Python, type hints are enforced using [mypy](https://github.com/python/mypy). This will be checked in CI.

## Python setup

If you want to contribute to the Python code, you also have to setup a Rust installation to be able to test your changes.
You have to follow these steps:

- install rust nightly via [rustup](https://www.rust-lang.org/tools/install)
- run `$ rustup override set nightly` from the root of the repo.
- from [./py-polars](./py-polars) run `$ pip3 install -r build.requirements.txt`
- **tests:** from [./py-polars](./py-polars) run `$ make test`
- **formatting + linting:** from [./py-polars](./py-polars) run `$ make pre-commit`

The last step installs a (slow) development build in your current environment and runs pytest.

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
