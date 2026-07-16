# HDFS

Polars on-premises provides experimental support for using HDFS (Hadoop Distributed Filesystem) as a
storage back-end, from a polars cluster. Accessing HDFS directly from the client is not included.
This functionality is considered **unstable**.

Polars HDFS support provides pure-Rust access to the data, no JVM required on the cluster. When
Iceberg is used, HDFS support also provides Rust-based native access to the metadata.

Support for accessing data and/or metadata requires explicit configuration, and may require
additional infrastructure to function.

Once enabled, HDFS storage is accessible through the standard polars API, using the
`hdfs://host:port/` URI syntax, in combination with the use of `storage_options`.

## Installation and Configuration

HDFS must be explicitly enabled in the cluster configuration, as follows.

```toml
[worker.extras]
hdfs.enabled = true
```

To access Iceberg metadata from HDFS, `pyiceberg` must also be enabled.

```toml
[worker.extras]
hdfs.enabled = true
pyiceberg.enabled = true
```

In addition, for metadata access, two python packages must be installed on the worker virtual
environment, i.e. `hdfs-native` and `pyiceberg`. This can be done by configuring the virtual
environment using your management tools, or as follows:

```python
pip install polars-on-premises[hdfs, pyiceberg]
```

Note, the python dependencies are not required for direct data access.

Under the hood, polars HDFS support uses the `hdfs_native` crate, which is responsible for handling
the Hadoop configuration, see [Native Rust HDFS client](https://github.com/kimahriman/hdfs-native)
for details.

By default, he HDFS client running on the worker will look for a configuration directory, which must
be made available locally on the worker. In the case of Kubernetes, this can be done by configuring
volume mounts in the helm chart as part of the deployment.

When Kerberos is used with the dynamic library `libgssapi-krb5-2`, the runtime image must include
this library, which will require an explicit runtime image update, or a ticket cache must be made
available locally from disk on every worker.

## Usage: Data access

Once enabled, direct HDFS data access is available using the standard polars API, for example:

```python
src = "hdfs://localhost:9005/data/foods1.parquet"

result = (
    pl.scan_parquet(src)
    .remote(context=ctx)
    .execute()
    .lazy()
    .collect()
)
```

Note that the URI contains both the scheme and the location.

In this case, no `storage_options` are required, but any `storage_options` provided will be shared
with the native Rust HDFS client, see
[Native Rust HDFS client](https://github.com/kimahriman/hdfs-native) for details.

## Usage: Iceberg metadata-path or table access

An Iceberg table with metadata stored on HDFS can be used for reading and writing by passing its
metadata path or by providing a table object.

The `storage_options` MUST include:

- the `py-io-impl` key with the value as shown below to select the no-JVM native Rust HDFS client,
  and
- the `hdfs.host` and optionally `hdfs.port` fields to the location, per the pyiceberg spec. The
  host/port from the URI will be ignored.

In future versions, polars will normalize the `storage_options` fields and URI elements to make the
API and configuration management more ergonomic.

With a known metadata path:

```python
storage_options = {
    "py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO",
    "hdfs.host": "localhost",
    "hdfs.port": "9005"
}

metadata_path = "hdfs://localhost:9005/warehouse/db/test/metadata/00001-a770f692-2344-4b87-b217-65b98ed7033b.metadata.json"

result = (
    pl.scan_iceberg(metadata_path, storage_options=storage_options)
    .remote(context=ctx)
    .execute()
    .lazy()
    .collect()
)
```

Similarly, a table object can be used:

```python
storage_options = {
    "py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO",
    "hdfs.host": "localhost",
    "hdfs.port": "9005"
}

(
    df.lazy()
    .remote(context=ctx)
    .distributed()
    .sink_iceberg(table, storage_options=storage_options, mode="append")
    .await_result()
)
```

Without `py-io-impl` set, pyiceberg will not function.

Note: accessing HDFS locally from the client is not supported. Only HDFS access from the cluster
nodes is currently supported.
