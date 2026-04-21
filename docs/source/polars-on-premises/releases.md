After obtaining a license for Polars on-premises by
[signing up here](https://w0lzyfh2w8o.typeform.com/to/zuoDgoMv) you will receive an access key for
our private Docker registry as well as a JSON-formatted license for running Polars on-premises.

## Binary

To pull the binary, run the following command:

```sh
curl -L 'https://get.onprem.pola.rs?version=<TAG>' --data @license.json --output polars-on-premises
```

### 0.2.4 (latest)

- `polars` [1.39.3](https://github.com/pola-rs/polars/releases/tag/py-1.39.3)
- `polars-cloud` [0.6.0](https://github.com/pola-rs/polars-cloud-client/releases/tag/client-0.6.0)

**Highlights**

- Single-node expression lowering
- Better query hanging detection
- Distributed slice
- Improved stage graph

### 0.2.3

- `polars` [1.39.3](https://github.com/pola-rs/polars/releases/tag/py-1.39.3)
- `polars-cloud` [0.6.0](https://github.com/pola-rs/polars-cloud-client/releases/tag/client-0.6.0)

**Highlights**

- Scratchpad
- Per-query historical profiling data in Control Plane
- Add config download and export query database endpoints
- Single-node lowering for memory-intensive operations
- Fix multi-partition bugs in the observatory (still one remaining for IO time)
- Add data skew and worker time information

### 0.1.1

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

## Helm chart

Helm chart releases are announced and documented on our
[chart repo](https://github.com/polars-inc/helm-charts/releases) directly. The version of the
embedded Polars on-premises binary is documented in the respective release summaries, or via a run
of the following command:

```sh
helm search repo polars-inc --versions
```

The container can be pulled from the DockerHub after logging in using `polarscustomer` as username
and the provided Personal Access Token (PAT) as password. The container images are tagged after the
versions listed above.
