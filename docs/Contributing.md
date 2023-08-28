# Contributing to the `Polars` User Guide

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it is:

- Reporting a bug.
- Discussing the current state of the code.
- Submitting a fix.
- Adding/proposing new features.

## General guide

### We develop with Github

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the [repo](https://github.com/pola-rs/polars-book.git) in your own GitHub account.
1. Clone your own version of the repo locally, and `cd` into it.
1. Add the upstream remote with `git remote add upstream https://github.com/pola-rs/polars-book.git`
1. All branches should derive from `master`, you can `git checkout -b <YOUR-BRANCH>` and write away.
1. Commit/push to your own repo.
1. Open a pull request as you would usually do, making sure the "base repository" is the upstream repo (`master` branch) and the "head repository" your own (`<YOUR-BRANCH>` branch).

To update your own repo with code pushed on the upstream repo:

1. `git checkout <BRANCH>`
1. `git pull upstream <BRANCH>`
1. `git push origin <BRANCH>`

### Building locally
To build the documentation locally you will need to install the python libraries defined in the `requirements.txt` file.

<!-- markdown-link-check-disable -->
When these steps are done run `mkdocs serve` to run the server. You can then view the docs at http://localhost:8000/
<!-- markdown-link-check-enable -->

### Want to discuss something?

Some questions will not fit an issue. For those we have [the Discord server](https://discord.gg/RhCg7uQCjQ).

### All contributions are under the MIT Software License

When contributing any content your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project.
Feel free to contact the maintainers if that is a concern.

### Report bugs using GitHub Issues

Do not hesitate to [open a new issue](https://github.com/pola-rs/polars-book/issues/new/choose).
**Great Issues** tend to have:

- A quick summary and/or background.
- Steps to reproduce.
- What you expected would happen.
- What actually happens.
- Notes (possibly including why you think this might be happening, or stuff you tried that did not work).

### Content formatting

The `Python` code is checked and linted using [`black`](https://github.com/psf/black). The recommended way is to run the black before commiting:

```shell
$ black .
```

## Code examples

Each time the User Guide is built, all examples are run against the latest release of `Polars` (as defined in the `requirements.txt` file found at the root of this repo).
To document a new functionality:

### Correct placement

Find the correct placement for the functionality. Is it an expression add it to the user-guide under the `docs/user-guide/expressions` folder, it is related to input / output than put it under the `docs/user-guide/IO` folder

The `Markdown` file should roughly match the following structure:

1. A clear short title (for example: "*Interact with an AWS bucket*").
1. A one-ish-liner to introduce the code snippet.
1. The code example itself under the corresponding folder (e.g. `docs/src/user-guide/expressions/...py), using the [Snippets](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/) syntax.
1. The output of the example, using [markdown-exec](https://pawamoy.github.io/markdown-exec/)
1. A longer explanation if required.
1. If applicable, provide both eager and lazy examples.

