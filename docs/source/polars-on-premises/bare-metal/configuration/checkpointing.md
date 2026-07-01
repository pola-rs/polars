# Checkpointing

When running long queries, you can enable checkpointing so that intermediate state is persisted
during execution. If a worker fails part way through, the query can resume from the last checkpoint
instead of starting over from the beginning.

Checkpointing requires configuration on both the scheduler and the workers: the scheduler decides
when checkpoints are created, while the workers store the checkpoint data.

!!! note

    Checkpointing is a no-op when the query's shuffles already write to a shared storage system. In
    that case the intermediate state is persisted there as part of normal execution, so there is
    nothing extra for checkpointing to store.

## Scheduler configuration

Enable checkpointing on the scheduler by adding the `[scheduler.checkpoint]` section. The `period`
controls how often checkpoints are created: once the period has passed after a stage has completed,
a checkpoint is created.

```toml
[scheduler]
enabled = true
# ...

[scheduler.checkpoint]
enabled = true
period = "10 mins"
```

Checkpointing must be turned on by setting `enabled = true`, just like the other components. The
`period` accepts either a jiff friendly duration format (see
[the jiff documentation](https://docs.rs/jiff/0.2.18/jiff/fmt/friendly/)) or an ISO 8601 duration
format, _e.g._ `PT10M` for 10 minutes.

## Worker configuration

Each worker needs a location to store its checkpoint data, configured through
`worker.checkpoint_location`. This can be a shared filesystem or an object store on S3, Google Cloud
Storage, or Azure Blob Storage.

### Shared filesystem

If your infrastructure has some shared storage file system, such as NFS (or CephFs, etc.), you can
use that here. The path must be accessible by all workers on the same path.

```toml
[worker]
enabled = true
checkpoint_location.shared_filesystem.path = "/mnt/storage/polars/checkpoints"
```

### S3-compatible storage

```toml
[worker]
enabled = true
checkpoint_location.s3.url = "s3://bucket/path/to/key"
checkpoint_location.s3.aws_access_key_id = "YOURACCESSKEY"
checkpoint_location.s3.aws_secret_access_key = "YOURSECRETKEY"
```

If you self-host an S3-compatible storage solution, you can override the `aws_endpoint_url`
configuration option.

```toml
[worker]
checkpoint_location.s3.aws_endpoint_url = "http://your-s3-compatible-storage-host:8080"
```

### Google Cloud Storage

```toml
[worker]
enabled = true
checkpoint_location.gcs.url = "gs://bucket/path/to/key"
checkpoint_location.gcs.google_service_account_path = "/etc/polars/gcs-service-account.json"
```

### Azure Blob Storage

```toml
[worker]
enabled = true
checkpoint_location.abs.url = "az://container/path/to/key"
checkpoint_location.abs.azure_storage_account_name = "YOURACCOUNT"
checkpoint_location.abs.azure_storage_account_key = "YOURKEY"
```

!!! note "Object store options"

    For the object store options (`s3`, `gcs`, and `abs`), the allowed keys are the same as in
    [`scan_parquet()`](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html)
    (_e.g._ `aws_access_key_id`, `google_service_account_path`, `azure_storage_account_name`). You
    can use any other cloud provider that supports the S3 API, such as MinIO or DigitalOcean Spaces.
