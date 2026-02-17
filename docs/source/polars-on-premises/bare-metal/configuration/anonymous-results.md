# Anonymous results

For remote Polars queries without a specific output sink, Polars on-premises can automatically add
an output sink. We call this sink an anonymous results sink. Infrastructure-wise, these sinks can be
backed by S3-compatible storage or another shared filesystem accessible from all worker nodes and
the Python client. The data written to this location is not automatically deleted, so you need to
configure a retention policy for this data yourself.

## Shared filesystem

If your infrastructure has some shared storage file system, such as NFS (or CephFs, etc.), you can
use that here. An example configuration is shown below:

```toml
[scheduler]
enabled = true
allow_local_sinks = true # required for local anonymous results
anonymous_results.local.path = "/mnt/storage/polars/anonymous-results"
```

Note that you must enable `allow_local_sinks` to allow query results to be written to a local path.

Make sure that this exact path is reachable from all worker nodes and the Python client. If the
Python client does not have access to this path, it won't be able to download the anonymous results,
but it will still be able to receive query status updates.

## S3 compatible storage

To store anonymous results in S3 compatible storage, you can configure it as shown below. The
credentials specified are automatically used in the worker. Once the anonymous results are written,
the scheduler also creates a presigned URL for the Python client to download the result from the S3
location.

```toml
[scheduler]
enabled = true
anonymous_results.s3.url = "s3://bucket/path/to/key"
anonymous_results.s3.aws_secret_access_key = "YOURSECRETKEY"
anonymous_results.s3.aws_access_key_id = "YOURACCESSKEY"
```

If you self-host an S3 compatible storage solution, you can override the `aws_endpoint_url`
configuration option.

```toml
[scheduler]
anonymous_results.s3.url = "s3://bucket/path/to/key"
anonymous_results.s3.aws_endpoint_url = "http://your-s3-compatible-storage-host:8080"
```

Make sure that this endpoint is reachable from all worker nodes and the Python client. If the Python
client does not have access to this endpoint, it won't be able to download the anonymous results,
but it will still be able to receive query status updates.

The allowed keys under `anonymous_results.s3` are the same as in
[`scan_parquet()`](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html)(_e.g._
`aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`, `aws_region`). We currently only
support the AWS keys of the `storage_options` dictionary, but note that you can use any other cloud
provider that supports the S3 API, such as MinIO or DigitalOcean Spaces.
