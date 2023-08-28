# Polars Book

This repo contains the [User Guide](https://pola-rs.github.io/polars-book/user-guide/index.html) for the [Polars DataFrame library](https://github.com/pola-rs/polars).

## Getting Started

The User guide is made with [Material for mkdocs](https://squidfunk.github.io/mkdocs-material/). In order to get started with building this book perform the following steps:

```shell
make requirements

```

In order to serve the books run `make serve`. This will run all the python examples and display the output inline using the `markdown-exec` plugin.

## Deployment

Deployment of the book is done using Github Pages and Github Workflows. The book is automatically deployed on each push to main branch. There are a number of checks in the CI pipeline to avoid non-working examples:

- Run all python examples, fail on any errors
- Run all node examples, fail on any errors
- Check all links in markdown
- Run black formatter
