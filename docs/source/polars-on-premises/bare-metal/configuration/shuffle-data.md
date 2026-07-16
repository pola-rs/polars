# Shuffle data

When running distributed queries, data needs to be transferred in between the nodes. Polars On-Prem
requires a configuration for this storage location. You should decide and benchmark which location
is the best for your infrastructure, as it has a large impact on query execution times.

## Worker local storage

When using local storage, Polars queries write shuffle data directly to a file. This is preferably
configured with a node local SSD. For other nodes to access the data, the following sequence
happens:

```
worker_1 -[fs write]-> disk
worker_2 <-[net]- worker_1 <-[fs read]- disk
```

Local shuffles can be configured as shown below:

```toml
[worker]
enabled = true
shuffle_location.local.path = "/mnt/storage/polars/shuffle"
# ...
```

## Worker shared storage

If your infrastructure has some shared storage file system, such as NFS (or CephFs, etc.), Polars
On-Prem can use that for its shuffle data too. This reduces shuffle complexity, as Polars can
directly write to the remote shared disk, and any worker can directly read from it. This setup can
lead to improved performance when the network storage provider is fast enough. In addition, it
provides automatic shuffle data persistence in case of worker node failure.

```
worker_1 -[net]-> shared storage -[fs]-> disk
worker_2 <-[net]- shared storage <-[fs]- disk
```

A requirement for this to work is that all workers have the same shuffle location configured. An
example configuration is shown below:

```toml
[worker]
enabled = true
shuffle_location.shared_filesystem.path = "/mnt/storage/polars/shuffle"
# ...
```

## S3-compatible storage

S3-compatible storage is similar to the shared filesystem storage described above, but uses the S3
API. It has the same advantages and disadvantages as the shared filesystem storage. You can
configure S3-compatible storage as follows:

```toml
[worker]
enabled = true
shuffle_location.s3.url = "s3://bucket/path/to/key"
shuffle_location.s3.aws_secret_access_key = "YOURSECRETKEY"
shuffle_location.s3.aws_access_key_id = "YOURACCESSKEY"
```

If you self-host an S3-compatible storage solution, you can override the `aws_endpoint_url`
configuration option.

```toml
[worker]
shuffle_location.s3.url = "s3://bucket/path/to/key"
shuffle_location.s3.aws_endpoint_url = "http://your-s3-compatible-storage-host:8080"
```

## Google Cloud Storage

Shuffle data can also be stored in Google Cloud Storage:

```toml
[worker]
enabled = true
shuffle_location.gcs.url = "gs://bucket/path/to/key"
shuffle_location.gcs.google_service_account_path = "/etc/polars/gcs-service-account.json"
```

## Azure Blob Storage

Shuffle data can also be stored in Azure Blob Storage:

```toml
[worker]
enabled = true
shuffle_location.abs.url = "az://container/path/to/key"
shuffle_location.abs.azure_storage_account_name = "YOURACCOUNT"
shuffle_location.abs.azure_storage_account_key = "YOURKEY"
```

!!! note "Object store options"

    For the object store options (`s3`, `gcs`, and `abs`), the allowed keys are the same as in
    [`scan_parquet()`](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html)
    (_e.g._ `aws_access_key_id`, `google_service_account_path`, `azure_storage_account_name`). You
    can use any other cloud provider that supports the S3 API, such as MinIO or DigitalOcean Spaces.
