## Scanning hive partitioned data

Polars supports scanning hive partitioned parquet and IPC datasets, with planned support for other
formats in the future.

Hive partition parsing is enabled by default if `scan_parquet` receives a single directory path,
otherwise it is disabled by default. This can be explicitly configured using the `hive_partitioning`
parameter.

### Scanning a hive directory

For this example the following directory structure is used:

```python exec="on" result="text" session="user-guide/io/hive"
--8<-- "python/user-guide/io/hive.py:init_paths"
```

Simply pass the directory to `scan_parquet`, and all files will be loaded with the hive parts in the
path included in the output:

{{code_block('user-guide/io/hive','scan_dir',['scan_parquet'])}}

```python exec="on" result="text" session="user-guide/io/hive"
--8<-- "python/user-guide/io/hive.py:scan_dir"
```

### Handling mixed files

Passing a directory to `scan_parquet` may not work if there are files with different extensions in
the directory.

For this example the following directory structure is used:

```python exec="on" result="text" session="user-guide/io/hive"
--8<-- "python/user-guide/io/hive.py:show_mixed_paths"
```

{{code_block('user-guide/io/hive','scan_dir_err',['scan_parquet'])}}

The above fails as `description.txt` is not a valid parquet file:

```python exec="on" result="text" session="user-guide/io/hive"
--8<-- "python/user-guide/io/hive.py:scan_dir_err"
```

In this situation, a glob pattern can be used to be more specific about which files to load. Note
that `hive_partitioning` must explicitly set to `True`:

{{code_block('user-guide/io/hive','scan_glob',['scan_parquet'])}}

```python exec="on" result="text" session="user-guide/io/hive"
--8<-- "python/user-guide/io/hive.py:scan_glob"
```

### Scanning file paths with hive parts

`hive_partitioning` is not enabled by default for file paths:

{{code_block('user-guide/io/hive','scan_file_no_hive',['scan_parquet'])}}

```python exec="on" result="text" session="user-guide/io/hive"
--8<-- "python/user-guide/io/hive.py:scan_file_no_hive"
```

Pass `hive_partitioning=True` to enable hive partition parsing:

{{code_block('user-guide/io/hive','scan_file_hive',['scan_parquet'])}}

```python exec="on" result="text" session="user-guide/io/hive"
--8<-- "python/user-guide/io/hive.py:scan_file_hive"
```

## Writing hive partitioned data

> Note: The following functionality is considered _unstable_, and is subject to change.

Polars supports writing hive partitioned parquet datasets, with planned support for other formats.

### Example

For this example the following DataFrame is used:

{{code_block('user-guide/io/hive','write_parquet_partitioned_show_data',[])}}

```python exec="on" result="text" session="user-guide/io/hive"
--8<-- "python/user-guide/io/hive.py:write_parquet_partitioned_show_data"
```

We will write it to a hive-partitioned parquet dataset, partitioned by the columns `a` and `b`:

{{code_block('user-guide/io/hive','write_parquet_partitioned',['write_parquet'])}}

```python exec="on" result="text" session="user-guide/io/hive"
--8<-- "python/user-guide/io/hive.py:write_parquet_partitioned"
```

The output is a hive partitioned parquet dataset with the following paths:

```python exec="on" result="text" session="user-guide/io/hive"
--8<-- "python/user-guide/io/hive.py:write_parquet_partitioned_show_paths"
```
