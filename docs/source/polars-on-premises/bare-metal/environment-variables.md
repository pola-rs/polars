These environment variables

| Variable                            | Description                                                                                                                                                                                                     |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POLARS_TEMP_DIR`                   | Override the default temporary directory Polars uses for scratch files and some I/O operations.                                                                                                                 |
| `PLC_LOG_LEVEL`                     | Controls logging verbosity for the polars-on-premises components (e.g. scheduler/worker).<br>e.g. Info`,`Debug`,`Trace`                                                                                         |
| `POLARS_SKIP_DSL_HASH_VERIFICATION` | Skips internal DSL schema/hash compatibility checks between polars cloud client and polars-on-premises binary.<br>Set to `1` only in controlled environments when working around temporary DSL hash mismatches. |
| `POLARS_ALLOW_PQ_EMPTY_STRUCT`      | Allows reading or writing Parquet files that contain empty struct fields that are otherwise rejected.                                                                                                           |
| `OTLP_ENDPOINT`                     | Target endpoint for sending OTLP traces/metrics/logs to your OpenTelemetry collector/observability stack.<br>e.g. `http://otel-collector:4317`                                                                  |
| `OTEL_SERVICE_INSTANCE_ID`          | OpenTelemetry `service.instance.id` that uniquely identifies this cublet instance in telemetry.<br>Must match `cublet_id`.                                                                                      |

You may also set any Polars OSS-recognized environment variables.
