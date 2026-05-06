# Anonymous results

For remote polars queries without a specific output sink, Polars Cloud can automatically add
persistent sink. We call these sinks "anonymous results" sinks. Infrastructure-wise, these sinks are
backed by S3-compatible storage, which should be accessible from all worker nodes and the python
client. The data written to this location is not automatically deleted, so you need to configure a
retention policy for this data yourself. You may configure the credentials as shown below. The key
names correspond to the
[`storage_options` parameter in `scan_parquet`](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html)
(e.g. `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`, `aws_region`). We currently
only support the AWS keys of the `storage_options` dictionary, but note that you can use any other
cloud provider that supports the S3 API, such as MinIO or DigitalOcean Spaces.

!!! note "Difference between Anonymous Users and Anonymous Results"

    Note that Anonymous Users and Anonymous Results are different. Anonymous Users refer to queries that are submitted without a username, while Anonymous Results refer to queries without an explicit output sink.

```yaml
anonymousResults:
  s3:
    enabled: true
    endpoint: "s3://my-bucket/path/to/dir"
    options:
      - name: aws_access_key_id
        valueFrom:
          secretKeyRef:
            name: my-s3-secret
            key: accessKeyId
      - name: aws_endpoint_url
        value: "http://localhost:9000"
  # etc.
```

If you wish to disable anonymous results, keep `anonymousResults.s3.enabled: false`. This will
ensure that all query result output locations need to be explicitly set by users.
