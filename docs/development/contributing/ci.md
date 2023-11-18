# Continuous integration

Polars uses GitHub Actions as its continuous integration (CI) tool. The setup is reasonably complex, as far as CI setups go. This page explains some of the design choices.

## Goal

Overall, the CI suite aims to achieve the following:

• Enforce code correctness by running automated tests.
• Enforce code quality by running automated linting checks.
• Enforce code performance by running benchmark tests.
• Enforce that code is properly documented.
• Allow maintainers to easily publish new releases.

We rely on a wide range of tools to achieve this for both the Rust and the Python code base, and thus a lot of checks are triggered on each pull request.

It's entirely possible that you submit a relatively trivial fix that subsequently fails a bunch of checks. Do not despair - check the logs to see what went wrong and try to fix it. You can run the failing command locally to verify that everything works correctly. If you can't figure it out, ask a maintainer for help!

## Design

The CI setup is designed with the following requirements in mind:

• Get feedback on each step individually. We want to avoid our test job being cancelled because a linting check failed, only to find out later that we also have a failing test.
• Get feedback on each check as quickly as possible. We want to be able to iterate quickly if it turns out our code does not pass some of the checks.
• Only run checks when they need to be run. A change to the Rust code does not warrant a linting check of the Python code, for example.

This results in a modular setup with many separate workflows and jobs that rely heavily on caching.

### Modular setup

The repository consists of two main parts: the Rust code base and the Python code base. Both code bases are interdependent: Rust code is tested through Python tests, and the Python code relies on the Rust implementation for most functionality.

To make sure CI jobs are only run when they need to be run, each workflow is triggered only when relevant files are modified.

### Caching

The main challenge is that the Rust code base for Polars is quite large, and consequently, compiling the project from scratch is slow. This is addressed by caching the Rust build artifacts.

However, since GitHub Actions does not allow sharing caches between feature branches, we need to run the workflows on the main branch as well - at least the part that builds the Rust cache. This leads to many workflows that trigger both on pull request AND on push to the main branch, with individual steps of jobs enabled or disabled based on the branch it runs on.

Care must also be taken not to exceed the maximum cache space of 10Gb allotted to open source GitHub repositories. Hence we do not do any caching on feature branches - we always use the cache available from the main branch. This also avoids any extra time that would be required to store the cache.

## Releases

The release jobs for Rust and Python are triggered manually.
Refer to the [contributing guide](./index.md#release-flow) for the full release process.
