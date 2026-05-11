# Profiling and host metrics

The profiling and host metrics system consists of a sender and receiving side. The observatory
component's main task is to receive telemetry data from the cluster's nodes. It exposes the
profiling data in a dashboard that can be accessed through the `polars-observatory` service.

## Observatory data stores

The observatory stores profiling data in an SQLite database file. By default, this file is stored in
an `emptyDir`, so data is lost when the scheduler pod is recreated. This volume can also be
configured as a persistent volume claim, allowing the observatory to persist profiling data across
scheduler pod restarts or even cluster recreation. The storage needed for this file is in the order
of several tens of MB, depending on the number of queries and their complexity.

You can configure the chart to create a persistent volume claim by configuring these volumes:

```yaml
observatory:
  data:
    persistentVolumeClaim:
      enabled: true
      storageClassName: "hostpath" # As configured in your k8s cluster
      size: 1Gi
```

Or reuse an existing volume claim as shown below. Make sure the volume has the `ReadWriteOnce`
access mode as the observatory requires exclusive access to the database file.

```yaml
observatory:
  data:
    persistentVolumeClaim:
      enabled: true
      create: false
      existingClaimName: "your-persistent-volume-claim"
```

The observatory stores host metrics in a pre-allocated buffer in memory. The size of this buffer can
be configured using the `observatory.maxMetricsBytesTotal` key. The default is around 100 MB. You
can estimate the required memory at around 50 bytes per second per worker node. So for a cluster
with 32 worker nodes, with a history of 1 hour, you would need around `50 * 32 * 3600 = 5760000`
bytes, or 5.7 MB.

## Host metrics exporter

The dashboard shows host metrics for each worker node. These metrics are exported by default and can
be disabled by setting `disableHostMetrics: true`.

The host metrics exporter supports collecting CPU and memory usage metrics from cgroups v1 and v2.

## Telemetry

Polars On-Prem uses OpenTelemetry as its telemetry framework. To receive OTLP metrics and traces,
configure `telemetry.otlpEndpoint` to point to your OTLP collector. Logs are written to stdout in
JSON format. For the compute plane, the log level can be configured using the `logLevel` value.
