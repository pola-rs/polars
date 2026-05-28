## 0.4.2 (latest)

- `polars` [1.40.1](https://github.com/pola-rs/polars/releases/tag/py-1.40.1)
- `polars-cloud` [0.7.0](https://github.com/pola-rs/polars-cloud-client/releases/tag/client-0.7.0)

**Highlights**

- Support for custom-provided environment variables
- Add distributed lowering for `Gather` and `RowIndexScans`
- Optimize shuffles and handle empty partitions
- Support GCS and ABS as anonymous result and/or shuffle locations

## 0.3.1

- `polars` [1.40.1](https://github.com/pola-rs/polars/releases/tag/py-1.40.1)
- `polars-cloud` [0.6.1](https://github.com/pola-rs/polars-cloud-client/releases/tag/client-0.6.1)

**Highlights**

- Apply pre-slice in scan row count estimation
- Distributed row index
- Implement lowering for row-index scans w/o predicates or pre-slices
- OpenLineage support
- Track shuffle outputs on the scheduler (this will later enable partial stage recovery)

## 0.2.4

- `polars` [1.39.3](https://github.com/pola-rs/polars/releases/tag/py-1.39.3)
- `polars-cloud` [0.6.0](https://github.com/pola-rs/polars-cloud-client/releases/tag/client-0.6.0)

**Highlights**

- Single-node expression lowering
- Better query hanging detection
- Distributed slice
- Improved stage graph

## 0.2.3

- `polars` [1.39.3](https://github.com/pola-rs/polars/releases/tag/py-1.39.3)
- `polars-cloud` [0.6.0](https://github.com/pola-rs/polars-cloud-client/releases/tag/client-0.6.0)

**Highlights**

- Scratchpad
- Per-query historical profiling data in Control Plane
- Add config download and export query database endpoints
- Single-node lowering for memory-intensive operations
- Fix multi-partition bugs in the observatory (still one remaining for IO time)
- Add data skew and worker time information

## 0.1.1

- `polars` [1.38.1](https://github.com/pola-rs/polars/releases/tag/py-1.38.1)
- `polars-cloud` [0.5.0](https://github.com/pola-rs/polars-cloud-client/releases/tag/client-0.5.0)

**Highlights**

- Streaming shuffles for local nodes (saves a full IO read + write during shuffles; this was already
  the case in shared-storage mode)
- Default to cost-based planner (much better stage graphs; unlocks more complicated plan lowering in
  next releases)
- IO metrics in `SCAN` in query plan profiler
- More distributed nodes: (stable sort, top-k, distinct)
- Pre-aggregate more aggregations: std, var, first, last
- Streaming ASOF joins

## Access the released artifacts

### Helm chart

Helm chart releases are announced and documented on our
[chart repo](https://github.com/polars-inc/helm-charts/releases) directly. The version of the
embedded Polars On-Prem binary is documented in the respective release summaries, or via a run of
the following command:

```sh
helm search repo polars-inc --versions
```

The container images are hosted on
[Dockerhub](https://hub.docker.com/r/polarscloud/polars-on-premises) and are tagged after the
versions listed above.

### Bare-metal binaries

After obtaining an offline license for Polars On-Prem you will receive a JSON-formatted license for
running Polars On-Prem.

To pull the binary, run the following command:

```sh
$ curl -L 'https://get.onprem.pola.rs?version=<TAG>' --data @license.json --output polars-on-premises
```

The versions follow the tags listed above.
