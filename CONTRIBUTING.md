# Contributing to Polars

Thanks for taking the time to contribute! We appreciate all contributions, from reporting bugs to implementing new features.
If you're unclear on how to proceed after reading this guide, please contact us on [Discord](https://discord.gg/4UfP5cfBE7).

## Table of contents

- [Reporting bugs](#reporting-bugs)
- [Suggesting enhancements](#suggesting-enhancements)
- [Contributing to the codebase](#contributing-to-the-codebase)
- [Contributing to documentation](#contributing-to-documentation)
- [Release flow](#release-flow)
- [License](#license)

## Reporting bugs

We use [GitHub issues](https://github.com/pola-rs/polars/issues) to track bugs and suggested enhancements.
You can report a bug by opening a [new issue](https://github.com/pola-rs/polars/issues/new/choose).
Use the appropriate issue type for the language you are using ([Rust](https://github.com/pola-rs/polars/issues/new?labels=bug&template=bug_report_rust.yml) / [Python](https://github.com/pola-rs/polars/issues/new?labels=bug&template=bug_report_python.yml)).

Before creating a bug report, please check that your bug has not already been reported, and that your bug exists on the latest version of Polars.
If you find a closed issue that seems to report the same bug you're experiencing, open a new issue and include a link to the original issue in your issue description.

Please include as many details as possible in your bug report. The information helps the maintainers resolve the issue faster.

## Suggesting enhancements

We use [GitHub issues](https://github.com/pola-rs/polars/issues) to track bugs and suggested enhancements.
You can suggest an enhancement by opening a [new feature request](https://github.com/pola-rs/polars/issues/new?labels=enhancement&template=feature_request.yml).
Before creating an enhancement suggestion, please check that a similar issue does not already exist.

Please describe the behavior you want and why, and provide examples of how Polars would be used if your feature were added.

## Contributing to the codebase

### Picking an issue

Pick an issue by going through the [issue tracker](https://github.com/pola-rs/polars/issues) and finding an issue you would like to work on.
Feel free to pick any issue with an [accepted](https://github.com/pola-rs/polars/issues?q=is%3Aopen+is%3Aissue+label%3Aaccepted) label that is not already assigned.
We use the [help wanted](https://github.com/pola-rs/polars/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) label to indicate issues that are high on our wishlist.

If you are a first time contributor, you might want to look for issues labeled [good first issue](https://github.com/pola-rs/polars/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).
The Polars code base is quite complex, so starting with a small issue will help you find your way around!

If you would like to take on an issue, please comment on the issue to let others know.
You may use the issue to discuss possible solutions.

### Setting up your local environment

Polars development flow relies on both Rust and Python, which means setting up your local development environment is not trivial.
If you run into problems, please contact us on [Discord](https://discord.gg/4UfP5cfBE7).

_Note that if you are a Windows user, the steps below might not work as expected; try developing using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)._

Start by [forking](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the Polars repository, then clone your forked repository using `git`:

```bash
git clone https://github.com/<username>/polars.git
cd polars
```

In order to work on Polars effectively, you will need [Rust](https://www.rust-lang.org/), [Python](https://www.python.org/), and [dprint](https://dprint.dev/).

First, install Rust using [rustup](https://www.rust-lang.org/tools/install).
After the initial installation, you will also need to install the nightly toolchain:

```bash
rustup toolchain install nightly --component miri
```

Next, install Python, for example using [pyenv](https://github.com/pyenv/pyenv#installation).
We recommend using the latest Python version (`3.11`).
Make sure you deactivate any active virtual environments or conda environments, as the steps below will create a new virtual environment for Polars.
You will need Python even if you intend to work on the Rust code only, as we rely on the Python tests to verify all functionality.

Finally, install [dprint](https://dprint.dev/install/).
This is not strictly required, but it is recommended as we use it to autoformat certain file types.

You can now check that everything works correctly by going into the `py-polars` directory and running the test suite
(warning: this may be slow the first time you run it):

```bash
cd py-polars
make test
```

This will do a number of things:

- Use Python to create a virtual environment in the `.venv` folder.
- Use [pip](https://pip.pypa.io/) to install all Python dependencies for development, linting, and building documentation.
- Use Rust to compile and install Polars in your virtual environment. _At least 8GB of RAM is recommended for this step to run smoothly._
- Use [pytest](https://docs.pytest.org/) to run the Python unittests in your virtual environment

Check if linting also works correctly by running:

```bash
make pre-commit
```

Note that we do not actually use the [pre-commit](https://pre-commit.com/) tool.
We use the Makefile to conveniently run the following formatting and linting tools:

- [black](https://black.readthedocs.io/) and [blackdoc](https://github.com/keewis/blackdoc)
- [ruff](https://github.com/charliermarsh/ruff)
- [mypy](http://mypy-lang.org/)
- [rustfmt](https://github.com/rust-lang/rustfmt)
- [clippy](https://doc.rust-lang.org/nightly/clippy/index.html)
- [dprint](https://dprint.dev/)

If this all runs correctly, you're ready to start contributing to the Polars codebase!

### Working on your issue

Create a new git branch from the `main` branch in your local repository, and start coding!

The Rust code is located in the `crates` directory, while the Python codebase is located in the `py-polars` directory.
Both directories contain a `Makefile` with helpful commands. Most notably:

- `make test` to run the test suite (see the [test suite docs](/py-polars/tests/README.md) for more info)
- `make pre-commit` to run autoformatting and linting

Note that your work cannot be merged if these checks fail!
Run `make help` to get a list of other helpful commands.

Two other things to keep in mind:

- If you add code that should be tested, add tests.
- If you change the public API, [update the documentation](#api-reference).

### Pull requests

When you have resolved your issue, [open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) in the Polars repository.
Please adhere to the following guidelines:

- Start your pull request title with a [conventional commit](https://www.conventionalcommits.org/) tag. This helps us add your contribution to the right section of the changelog. We use the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#type). Scope can be `rust` and/or `python`, depending on your contribution.
- Use a descriptive title. This text will end up in the [changelog](https://github.com/pola-rs/polars/releases).
- In the pull request description, [link](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) to the issue you were working on.
- Add any relevant information to the description that you think may help the maintainers review your code.
- Make sure your branch is [rebased](https://docs.github.com/en/get-started/using-git/about-git-rebase) against the latest version of the `main` branch.
- Make sure all [GitHub Actions checks](/.github/workflows/README.md) pass.

After you have opened your pull request, a maintainer will review it and possibly leave some comments.
Once all issues are resolved, the maintainer will merge your pull request, and your work will be part of the next Polars release!

Keep in mind that your work does not have to be perfect right away!
If you are stuck or unsure about your solution, feel free to open a draft pull request and ask for help.

## Contributing to documentation

The most important components of Polars documentation are the [user guide](https://pola-rs.github.io/polars/user-guide/), the API references, and the database of questions on [StackOverflow](https://stackoverflow.com/).

### User guide

The user guide is maintained in the `docs/user-guide` folder. Before creating a PR first raise an issue to discuss what you feel is missing or could be improved.

#### Building and serving the user guide

The user guide is built using [MkDocs](https://www.mkdocs.org/). You install the dependencies for building the user guide by running `make requirements` in the root of the repo.

Run `mkdocs serve` to build and serve the user guide, so you can view it locally and see updates as you make changes.

#### Creating a new user guide page

Each user guide page is based on a `.md` markdown file. This file must be listed in `mkdocs.yml`.

#### Adding a shell code block

To add a code block with code to be run in a shell with tabs for Python and Rust, use the following format:

````
=== ":fontawesome-brands-python: Python"

    ```shell
    $ pip install fsspec
    ```

=== ":fontawesome-brands-rust: Rust"

    ```shell
    $ cargo add aws_sdk_s3
    ```
````

#### Adding a code block

The snippets for Python and Rust code blocks are in the `docs/src/python/` and `docs/src/rust/` directories, respectively. To add a code snippet with Python or Rust code to a `.md` page, use the following format:

```
{{code_block('user-guide/io/cloud-storage','read_parquet',[read_parquet,read_csv])}}
```

- The first argument is a path to either or both files called `docs/src/python/user-guide/io/cloud-storage.py` and `docs/src/rust/user-guide/io/cloud-storage.rs`.
- The second argument is the name given at the start and end of each snippet in the `.py` or `.rs` file
- The third argument is a list of links to functions in the API docs. For each element of the list there must be a corresponding entry in `docs/_build/API_REFERENCE_LINKS.yml`

If the corresponding `.py` and `.rs` snippet files both exist then each snippet named in the second argument to `code_block` above must exist or the build will fail. An empty snippet should be added to the `.py` or `.rs` file if the snippet is not needed.

Each snippet is formatted as follows:

```python
# --8<-- [start:read_parquet]
import polars as pl

df = pl.read_parquet("file.parquet")
# --8<-- [end:read_parquet]
```

The snippet is delimited by `--8<-- [start:<snippet_name>]` and `--8<-- [end:<snippet_name>]`. The snippet name must match the name given in the second argument to `code_block` above.

#### Linting

Before committing, install `dprint` (see above) and run `dprint fmt` from the `docs` directory to lint the markdown files.

### API reference

Polars has separate API references for [Rust](https://pola-rs.github.io/polars/docs/rust/dev/polars/) and [Python](https://pola-rs.github.io/polars/docs/python/dev/reference/index.html).
These are generated directly from the codebase, so in order to contribute, you will have to follow the steps outlined in [this section](#contributing-to-the-codebase) above.

#### Rust

Rust Polars uses `cargo doc` to build its documentation. Contributions to improve or clarify the API reference are welcome.

#### Python

For the Python API reference, we always welcome good docstring examples.
There are still parts of the API that do not have any code examples.
This is a great way to start contributing to Polars!

Note that we follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) convention.
Docstring examples should also follow the [Black](https://black.readthedocs.io/) codestyle.
From the `py-polars` directory, run `make fmt` to make sure your additions pass the linter, and run `make doctest` to make sure your docstring examples are valid.

Polars uses Sphinx to build the API reference.
This means docstrings in general should follow the [reST](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) format.
If you want to build the API reference locally, go to the `py-polars/docs` directory and run `make html SPHINXOPTS=-W`.
The resulting HTML files will be in `py-polars/docs/build/html`.

New additions to the API should be added manually to the API reference by adding an entry to the correct `.rst` file in the `py-polars/docs/source/reference` directory.

### StackOverflow

We use StackOverflow to create a database of high quality questions and answers that is searchable and remains up-to-date.
There is a separate tag for each language:

- [Python Polars](https://stackoverflow.com/questions/tagged/python-polars)
- [Rust Polars](https://stackoverflow.com/questions/tagged/rust-polars)

Contributions in the form of well-formulated questions or answers are always welcome!
If you add a new question, please notify us by adding a [matching issue](https://github.com/pola-rs/polars/issues/new?&labels=question&template=question.yml) to our GitHub issue tracker.

## Release flow

_This section is intended for Polars maintainers._

Polars releases Rust crates to [crates.io](https://crates.io/crates/polars) and Python packages to [PyPI](https://pypi.org/project/polars/).

New releases are marked by an official [GitHub release](https://github.com/pola-rs/polars/releases) and an associated git tag. We utilize [Release Drafter](https://github.com/release-drafter/release-drafter) to automatically draft GitHub releases with release notes.

### Steps

The steps for releasing a new Rust or Python version are similar. The release process is mostly automated through GitHub Actions, but some manual steps are required. Follow the steps below to release a new version.

Start by bumping the version number in the source code:

1. Check the [releases page](https://github.com/pola-rs/polars/releases) on GitHub and find the appropriate draft release. Note the version number associated with this release.
2. Make sure your fork is up-to-date with the latest version of the main Polars repository, and create a new branch.
3. Bump the version number.

- _Rust:_ Update the version number in all `Cargo.toml` files in the `polars` directory and subdirectories. You'll probably want to use some search/replace strategy, as there are quite a few crates that need to be updated.
- _Python:_ Update the version number in [`py-polars/Cargo.toml`](https://github.com/pola-rs/polars/blob/main/py-polars/Cargo.toml#L3) to match the version of the draft release.

4. From the `py-polars` directory, run `make build` to generate a new `Cargo.lock` file.
5. Create a new commit with all files added. The name of the commit should follow the format `release(<language>): <Language> Polars <version-number>`. For example: `release(python): Python Polars 0.16.1`
6. Push your branch and open a new pull request to the `main` branch of the main Polars repository.
7. Wait for the GitHub Actions checks to pass, then squash and merge your pull request.

Directly after merging your pull request, release the new version:

8. Go to the release workflow ([Python](https://github.com/pola-rs/polars/actions/workflows/release-python.yml)/[Rust](https://github.com/pola-rs/polars/actions/workflows/release-rust.yml)), click _Run workflow_ in the top right, and click the green button. This will trigger the workflow, which will build all release artifacts and publish them.
9. Wait for the workflow to finish, then check [crates.io](https://crates.io/crates/polars)/[PyPI](https://pypi.org/project/polars/)/[GitHub](https://github.com/pola-rs/polars/releases) to verify that the new Polars release is now available.

### Troubleshooting

It may happen that one or multiple release jobs fail. If so, you should first try to simply re-run the failed jobs from the GitHub Actions UI.

If that doesn't help, you will have to figure out what's wrong and commit a fix. Once your fix has made it to the `main` branch, simply re-trigger the release workflow.

## License

Any contributions you make to this project will fall under the [MIT License](LICENSE) that covers the Polars project.
