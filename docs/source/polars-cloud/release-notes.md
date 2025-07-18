# Release Notes

## Polars Cloud 0.0.7 (2025-04-11)

### Enhancements

- We now default to the streaming engine for distributed workloads A big benefit is that query
  results are now streamed directly to final S3/external storage instead of being collected into
  memory first. This enables handling much larger results without causing out-of-memory errors.
- Polars Cloud now supports organizations. An organization can contain multiple workspaces and will
  eventually serve as the central entity for managing user roles and billing. During migration, an
  organization was automatically created for each of your workspaces. If you have multiple
  workspaces with the same name, you can now select one by specifying the organization:
  ```python
  workspace = Workspace("my_workspace", organization="my_organization")
  context = ComputeContext(workspace=workspace)
  ```
- Polars Cloud login tokens are now automatically refreshed for up to 8 hours after the initial
  login. No more need to login every 15 minutes.
- A clear error message is now shown when connecting to Polars Cloud with an out-of-date client
  version.
- The portal now has a feedback button in the bottom right corner so you can easily report feedback
  to us. We greatly appreciate it, so please use it as much as you would like.
- Distributed queries now display the physical plan in addition to the logical plan. The logical
  plan is the optimized query and how it would be run on a single node. The physical plan represents
  how we chop this up into stages and tasks so that it can be run on multiple nodes in a distributed
  manner.
- The workspace compute listing page has been reworked for improved clarity and shows more
  information.

### Bug fixes

- Improved out-of-memory reporting
- Ctrl-C now works to exit during login
- Workspace members can now be re-invited if they were previously removed

## Polars Cloud 0.0.6 (2025-03-31)

This version of Polars Cloud is only compatible with the 1.26.x releases of Polars Open Source.

### Enhancements

- Added support for [Polars IO plugins](https://docs.pola.rs/user-guide/plugins/io_plugins/).
  - These allow you to register different file formats as sources to the Polars engines which allows
    you to benefit from optimizations like projection pushdown, predicate pushdown, early stopping
    and support of our streaming engine.
- Added `.show()` for remote queries
  - You can now call `.remote().show()` instead of `remote().limit().collect().collect()`
- All Polars features that rely on external Python dependencies are now available in Polars Cloud
  such as:
  - `scan_iceberg()`
  - `scan_delta()`
  - `read_excel()`
  - `read_database()`

### Bug fixes

- Fixed an issue where specifying CPU/memory minimums could select an incompatible legacy EC2
  instance.
- The API now returns clear errors when trying to start a query on an uninitialized workspace.
- Fixed a 404 error when opening a query overview page in a workspace different from the currently
  selected workspace.
- Viewing another workspace members compute detail page no longer shows a blank screen.
