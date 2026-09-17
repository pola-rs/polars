# Query monitoring

Polars can collect metrics while your queries run and send them to
[Polars Cloud](https://cloud.pola.rs/). Each run shows up as a
[query profile](https://docs.cloud.pola.rs/polars-cloud/run/query-profile/) in the dashboard, which
shows you which part of a query is slow. That page explains what the metrics mean.

Monitoring works for queries that run locally; you do not need to move your data or your compute to
Polars Cloud to use it. Your data stays on your machine. No rows or column values are transmitted,
only the query plan and per-node runtime counters. The plan includes the schema, column names,
source paths, and any literals you wrote into the query.

## Requirements

- A Polars Cloud account. Get started at
  [Polars Cloud](https://docs.cloud.pola.rs/polars-cloud/get-started/).
- The `polars-cloud` package in the same environment as Polars:

```bash
pip install 'polars-cloud>=0.11.0'
```

## Enabling monitoring

```python
import polars as pl

pl.Config.enable_monitoring()
```

The first call opens a browser window to authenticate your session if you are not already logged in.
Every query you collect afterward reports its metrics to your
[Polars Cloud dashboard](https://cloud.pola.rs/portal).

Use the streaming engine when monitoring. It is the only engine that reports per-node metrics, so
enabling monitoring also sets the engine affinity to `"streaming"`.

## Choosing a workspace

Metrics go to the default workspace of your account. Pass `workspace` to send them somewhere else,
and `organization` to disambiguate a workspace name that exists in more than one organization:

```python
pl.Config.enable_monitoring(workspace="My workspace")
pl.Config.enable_monitoring(workspace="My workspace", organization="My organization")
```

## Monitoring a single query

Monitoring usually stays on for the whole session. If you want it for one query only, `Config` also
works as a context manager: the previous monitoring state _and_ the previous engine affinity are
restored on exit.

```python
with pl.Config(enable_monitoring=True):
    lf.collect()
```

## Disabling monitoring

```python
pl.Config.enable_monitoring(False)
```
