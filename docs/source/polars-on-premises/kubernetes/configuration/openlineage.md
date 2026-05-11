# OpenLineage

OpenLineage is an open platform for collection and analysis of data lineage. See
[openlineage.io](https://openlineage.io) for more information. See
[OpenLineage Integration](https://docs.pola.rs/polars-on-premises/integrations/openlineage/) for
more information on how to annotate Polars queries and inspecting them in an OpenLineage collector.

The cluster must be configured with a lineage transport endpoint, pointing at the collector. HTTP(S)
is the only supported transport protocol. The following example points at an instance of Marquez
deployed in the default namespace:

```yaml
lineage:
  enabled: true
  transport:
    http:
      endpoint: "http://marquez.default.svc.cluster.local:5000"
```

See [OpenLineage Integration](/polars-on-premises/integrations/openlineage) for more information on
how to annotate Polars queries and inspect them in an OpenLineage collector.
