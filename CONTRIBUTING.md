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
  4. Ensure the test suite passes:
     ```bash
     # For Rust code in ./polars in subdir
     cd ./polars
     make test

     # For Python code and Rust code in ./py-polars subdir.
     cd ./py-polars
     make test
     ```
  5. Make sure your code lints:
     ```bash
     # For Rust code in ./polars subdir.
     #   - cargo clippy: Lint Rust code (./polars/).
     #   - cargo fmt: Format Rust code (./polars/).
     #   - dprint fmt: Format TOML files.
     cd ./polars
     make clippy
     make fmt

     # For Python code and rust code in ./py-polars subdir:
     #   - isort: Sort Python imports.
     #   - black: Format Python code.
     #   - blackdoc: Format Python doctests.
     #   - mypy: Type checking of Python code.
     #   - flake8: Enforce Python style guide.
     #   - dprint fmt: Format TOML files.
     #   - cargo fmt: Format Rust code (./py-polars/src).
     cd ./py-polars
     make pre-commit
     ```
  6. Issue that pull request!


## Want to discuss something?

I can imagine that some questions don't fit an issue.
Therefore there is also a [Polars Discord server](https://discord.gg/4UfP5cfBE7).


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

  - **Rust** code with:
      - [cargo fmt](https://rust-lang.github.io/)
          * In `./polars subdir`: `$ cargo fmt --all`
          * In `./py-polars subdir` for Rust Python-bindings: `$ cargo fmt --all`

  - **Python** code with:
      - [isort](https://github.com/PyCQA/isort) (version 5.9.2):
          * Sort Python imports.
          * `isort .`
      - [black](https://github.com/psf/black) (version 21.6b0):
          * Format Python code.
          * `black .`

  - **Python** code in doctests:
      - [blackdoc](https://blackdoc.readthedocs.io/en/latest/) (version 0.3.4):
          * `blackdoc .`

  - **TOML** files with:
      - [dprint](https://github.com/dprint/dprint) (version 0.18.2):
          * `$ dprint fmt`

See `5. Make sure your code lints` for running it easily.


## Linting

We use linters to enforce code quality. This will be checked in CI.

  - **Rust**:
      - [clippy](https://github.com/rust-lang/rust-clippy) as Rust linter.

  - **Python**:
      - [flake8](https://flake8.pycqa.org/en/latest/) as Python linter.

See `5. Make sure your code lints` for running it easily.


## Type checking

For Python, type hints are enforced using [mypy](https://github.com/python/mypy). This will be checked in CI.

See `5. Make sure your code lints.` for running it easily.


## Testing

See `4. Ensure the test suite passes` for running it easily.


## Python setup

If you want to contribute to the Python code, you also have to setup a Rust installation to be able to test your changes.
You have to follow these steps:

  - Install Rust nightly via [rustup](https://www.rust-lang.org/tools/install)
  - run `$ rustup override set nightly` from the root of the repo.
  - from [./py-polars](./py-polars) run `$ pip3 install -r build.requirements.txt`
  - **tests:** from [./py-polars](./py-polars) run `$ make test`
  - **formatting + linting:** from [./py-polars](./py-polars) run `$ make pre-commit` before committing.

`make test` installs a (slow) development build in your current environment and runs `pytest`.


## License

By contributing, you agree that your contributions will be licensed under its MIT License.
