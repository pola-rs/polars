# Polars test suite

This folder contains the main Polars test suite. This document contains some information on the various components of the test suite, as well as guidelines for writing new tests.

The test suite contains four main components, each confined to their own folder: unit tests, parametric tests, benchmark tests, and doctests.

Note that this test suite is indirectly responsible for testing Rust Polars as well. The Rust test suite is kept small to reduce compilation times. A lot of the Rust functionality is tested here instead.

## Table of contents

- [Unit tests](#unit-tests)
- [Parametric tests](#parametric-tests)
- [Doctests](#doctests)
- [Benchmark tests](#benchmark-tests)

## Unit tests

The `unit` folder contains all regular unit tests.
These tests are intended to make sure all Polars functionality works as intended.

### Running unit tests

Run unit tests by running `make test` from the `py-polars` folder. This will compile the Rust bindings and then run the unit tests.

If you're working in the Python code only, you can avoid recompiling every time by simply running `pytest` instead.

By default, slow tests are skipped. Slow tests are marked as such using a [custom pytest marker](https://docs.pytest.org/en/latest/example/markers.html).
If you wish to run slow tests, run `pytest -m slow`.
Or run `pytest -m ""` to run _all_ tests, regardless of marker.

Tests can be run in parallel using [`pytext-xdist`](https://pytest-xdist.readthedocs.io/en/latest/). Run `pytest -n auto` to parallelize your test run.

### Writing unit tests

Whenever you add new functionality, you should also add matching unit tests.
Add your tests to appropriate test module in the `unit` folder.
Some guidelines to keep in mind:

- Try to fully cover all possible inputs and edge cases you can think of.
- Utilize pytest tools like [`fixture`](https://docs.pytest.org/en/latest/explanation/fixtures.html) and [`parametrize`](https://docs.pytest.org/en/latest/how-to/parametrize.html) where appropriate.
- Since many tests will require some data to be defined first, it can be efficient to run multiple checks in a single test. This can also be addressed using pytest fixtures.
- Unit tests should not depend on external factors, otherwise test parallelization will break.

## Parametric tests

The `parametric` folder contains parametric tests written using the [Hypothesis](https://hypothesis.readthedocs.io/) framework.
These tests are intended to find and test edge cases by generating many random datapoints.

### Running parametric tests

Run parametric tests by running `pytest -m hypothesis`.

Note that parametric tests are excluded by default when running `pytest`.
You must explicitly specify `-m hypothesis` to run them.

These tests _will_ be included when calculating test coverage, and will also be run as part of the `make test-all` make command.

## Doctests

The `docs` folder contains a script for running [`doctest`](https://docs.python.org/3/library/doctest.html).
This folder does not contain any actual tests - rather, the script checks all docstrings in the Polars package for `Examples` sections, runs the code examples, and verifies the output.

The aim of running `doctest` is to make sure the `Examples` sections in our docstrings are valid and remain up-to-date with code changes.

### Running `doctest`

To run the `doctest` module, run `make doctest` from the `py-polars` folder.
You can also run the script directly from your virtual environment.

Note that doctests are _not_ run using pytest. While pytest does have the capability to run doc examples, configuration options are too limited for our purposes.

Doctests will _not_ count towards test coverage. They are not a substitute for unit tests, but rather intended to convey the intended use of the Polars API to the user.

### Writing doc examples

Almost all classes/methods/functions that are part of Polars' public API should include code examples in their docstring.
These examples help users understand basic usage and allow us to illustrate more advanced concepts as well.
Some guidelines for writing a good docstring `Examples` section:

- Start with a minimal example that showcases the default functionality.
- Showcase the effect of its parameters.
- Showcase any special interactions when combined with other code.
- Keep it succinct and avoid multiple examples showcasing the same thing.

There are many great docstring examples already, just check other code if you need inspiration!

In addition to the [regular options](https://docs.python.org/3/library/doctest.html#option-flags) available when writing doctests, the script configuration allows for a new `IGNORE_RESULT` directive. Use this directive if you want to ensure the code runs, but the output may be random by design or not interesting to check.

```python
>>> df.sample(n=2)  # doctest: +IGNORE_RESULT
```

## Benchmark tests

The `benchmark` folder contains code for running the [H2O AI database benchmark](https://github.com/h2oai/db-benchmark).
It also contains various other benchmark tests.

The aim of this part of the test suite is to spot performance regressions in the code, and to verify that Polars functionality works as expected when run on a release build or at a larger scale.

### Running the H2O AI database benchmark

The benchmark is somewhat cumbersome to run locally. You must first generate the dataset using the R script provided in the `benchmark` folder. Afterwards, you can simply run the Python script to run the benchmark.

Make sure to install a release build of Polars before running the benchmark to guarantee the best results.

Refer to the [benchmark workflow](/.github/workflows/benchmark.yaml) for detailed steps.

### Running other benchmark tests

The other benchmark tests are run using pytest.
Run `pytest -m benchmark --durations 0 -v` to run these tests and report run duration.

Note that benchmark tests are excluded by default when running `pytest`.
You must explicitly specify `-m benchmark` to run them.
They will also be excluded when calculating test coverage.

These tests _will_ be run as part of the `make test-all` make command.
