# Contributing to Polars
We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Adding/Proposing new features

## We Develop with Github
We use github to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Want to discuss something?
I can image that some questions don't fit an issue. 
Therefore there is also a [chat on gitter](https://gitter.im/polars-rs/community).

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same 
[MIT License](http://choosealicense.com/licenses/mit/) that covers the project. 
Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/ritchie46/polars/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue]().
**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Code formatting
We test the code formatting in the CI pipelines. If you don't want these to fail, you need to format:

* **Rust** code with `$ cargo fmt`
* **Python** code with [black (version 20.8b1)](https://github.com/psf/black), running `$ black .`

## Linting
We use linters to enforce code quality. This will be checked in CI.

* **Rust** We use [clippy](https://github.com/rust-lang/rust-clippy) as linter. 
* **Python** We use [flake8](https://flake8.pycqa.org/en/latest/) as linter. 

## License
By contributing, you agree that your contributions will be licensed under its MIT License.
