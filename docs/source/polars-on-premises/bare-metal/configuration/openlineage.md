# OpenLineage

OpenLineage is an open platform for collection and analysis of data lineage. See
[openlineage.io](https://openlineage.io) for more information. See
[OpenLineage Integration](/polars-on-premises/integrations/openlineage) for more information on how
to annotate Polars queries and inspecting them in an OpenLineage collector.

## Configuration

The cluster must be configured with a lineage transport endpoint, pointing at the collector. HTTP(S)
is the only supported transport protocol. The following example points at a local instance of
Marquez.

```toml
[lineage]
enabled = true
transport.http.endpoint = "http://localhost:5000"
```

See [OpenLineage Integration](/polars-on-premises/integrations/openlineage) for more information on
how to annotate Polars queries and inspect them in an OpenLineage collector.
