# Shuffle data

When running distributed queries, data needs to be transferred in between the nodes. Polars
On-Prem requires a configuration for this storage location. You should decide and benchmark
which location is the best for your infrastructure, as it has a large impact on query execution
times.

## Worker local storage

When using local storage, Polars queries write shuffle data directly to a file. This is preferably
configured with a node local SSD. For other nodes to access the data, the following sequence
happens:

```
worker_1 -[fs write]-> disk
worker_2 <-[net]- worker_1 <-[fs read]- disk
```

By default, the ephemeral volume for this is disabled, and an `emptyDir` volume is used instead.
However, to prevent the host from running out of disk space during large queries, it is recommended
to enable an ephemeral volume for this purpose. The feature below will add a
[Generic Ephemeral Volume](https://kubernetes.io/docs/concepts/storage/ephemeral-volumes/) to each
of the pods.

```yaml
shuffleData:
  ephemeralVolumeClaim:
    enabled: true
    storageClassName: "hostpath" # As configured in your k8s cluster
    size: 125Gi
```

## Worker shared storage

If your infrastructure has some shared storage file system, such as NFS (or CephFs, etc.), Polars
On-Prem can use that for its shuffle data too. In Kubernetes terms, this boils down to having a
fast `ReadWriteMany` capable storage class. This shuffle type reduces complexity, as Polars can
directly write to the shared disk, and any worker can directly read from it. This setup can lead to
improved performance when the network storage provider is fast enough. In addition, it provides
automatic shuffle data persistence in case of worker node failure.

```
worker_1 -[net]-> shared storage -[fs]-> disk
worker_2 <-[net]- shared storage <-[fs]- disk
```

Configure it as show below:

```yaml
shuffleData:
  sharedPersistentVolumeClaim:
    enabled: true
    storageClassName: "cephfs" # As configured in your k8s cluster
    size: 125Gi
```

## S3 compatible storage

S3 compatible storage is similar to the shared filesystem storage described above, but uses the S3
API. It has the same advantages and disadvantages as the shared filesystem storage. You may
configure the credentials as shown below. The key names correspond to the
[`storage_options` parameter in `scan_parquet`](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html)
(e.g. `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`, `aws_region`). We currently
only support the AWS keys of the `storage_options` dictionary, but note that you can use any other
cloud provider that supports the S3 API, such as MinIO or DigitalOcean Spaces.

```yaml
shuffleData:
  s3:
    enabled: true
    endpoint: "s3://my-bucket/path/to/dir"
    options:
      - name: aws_access_key_id
        valueFrom:
          secretKeyRef:
            name: my-s3-secret
            key: accessKeyId
  # etc.
```

If you self-host an S3 compatible storage solution, you can override the `aws_endpoint_url`
configuration option.

```yaml
shuffleData:
  s3:
    enabled: true
    endpoint: "s3://my-bucket/path/to/dir"
    options:
      - name: aws_endpoint_url
        value: "http://your-s3-compatible-storage-host:8080"
  # etc.
```
