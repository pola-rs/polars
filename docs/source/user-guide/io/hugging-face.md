# Hugging Face

## Scanning datasets from Hugging Face

All cloud-enabled scan functions, and their `read_` counterparts transparently support scanning from
Hugging Face:

| Scan                                                                                          | Read                                                                                          |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [scan_parquet](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html) | [read_parquet](https://docs.pola.rs/api/python/stable/reference/api/polars.read_parquet.html) |
| [scan_csv](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_csv.html)         | [read_csv](https://docs.pola.rs/api/python/stable/reference/api/polars.read_csv.html)         |
| [scan_ndjson](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_ndjson.html)   | [read_ndjson](https://docs.pola.rs/api/python/stable/reference/api/polars.read_ndjson.html)   |
| [scan_ipc](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_ipc.html)         | [read_ipc](https://docs.pola.rs/api/python/stable/reference/api/polars.read_ipc.html)         |

### Path format

To scan from Hugging Face, a `hf://` path can be passed to the scan functions. The `hf://` path format is defined as `hf://BUCKET/REPOSITORY@REVISION/PATH`, where:

- `BUCKET` is one of `datasets` or `spaces`
- `REPOSITORY` is the location of the repository, this is usually in the format of `username/repo_name`. A branch can also be optionally specified by appending `@branch`
- `REVISION` is the name of the branch (or commit) to use. This is optional and defaults to `main` if not given.
- `PATH` is a file or directory path, or a glob pattern from the repository root.

Example `hf://` paths:

| Path                                                  | Path components                                                                                                                                                                                 |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| hf://datasets/nameexhaustion/polars-docs/iris.csv     | Bucket: datasets<br>Repository: nameexhaustion/polars-docs<br>Branch: main<br>Path: iris.csv<br> [Web URL](https://huggingface.co/datasets/nameexhaustion/polars-docs/tree/main/)               |
| hf://datasets/nameexhaustion/polars-docs@foods/\*.csv | Bucket: datasets<br>Repository: nameexhaustion/polars-docs<br>Branch: foods<br>Path: \*.csv<br> [Web URL](https://huggingface.co/datasets/nameexhaustion/polars-docs/tree/foods/)               |
| hf://datasets/nameexhaustion/polars-docs/hive_dates/  | Bucket: datasets<br>Repository: nameexhaustion/polars-docs<br>Branch: main<br>Path: hive_dates/<br> [Web URL](https://huggingface.co/datasets/nameexhaustion/polars-docs/tree/main/hive_dates/) |
| hf://spaces/nameexhaustion/polars-docs/orders.feather | Bucket: spaces<br>Repository: nameexhaustion/polars-docs<br>Branch: main<br>Path: orders.feather<br> [Web URL](https://huggingface.co/spaces/nameexhaustion/polars-docs/tree/main/)             |

### Authentication

A Hugging Face API key can be passed to Polars to access private locations using either of the following methods:

- Passing a `token` in `storage_options` to the scan function, e.g. `scan_parquet(..., storage_options={'token': '<your HF token>'})`
- Setting the `HF_TOKEN` environment variable, e.g. `export HF_TOKEN=<your HF token>`

### Examples

#### CSV

```python exec="on" result="text" session="user-guide/io/hugging-face"
--8<-- "python/user-guide/io/hugging-face.py:setup"
```

{{code_block('user-guide/io/hugging-face','scan_iris_csv',['scan_csv'])}}

```python exec="on" result="text" session="user-guide/io/hugging-face"
--8<-- "python/user-guide/io/hugging-face.py:scan_iris_repr"
```

See this file at [https://huggingface.co/datasets/nameexhaustion/polars-docs/blob/main/iris.csv](https://huggingface.co/datasets/nameexhaustion/polars-docs/blob/main/iris.csv)

#### NDJSON

{{code_block('user-guide/io/hugging-face','scan_iris_ndjson',['scan_ndjson'])}}

```python exec="on" result="text" session="user-guide/io/hugging-face"
--8<-- "python/user-guide/io/hugging-face.py:scan_iris_repr"
```

See this file at [https://huggingface.co/datasets/nameexhaustion/polars-docs/blob/main/iris.jsonl](https://huggingface.co/datasets/nameexhaustion/polars-docs/blob/main/iris.jsonl)

#### Parquet

{{code_block('user-guide/io/hugging-face','scan_parquet_hive_repr',['scan_parquet'])}}

```python exec="on" result="text" session="user-guide/io/hugging-face"
--8<-- "python/user-guide/io/hugging-face.py:scan_parquet_hive_repr"
```

See this folder at [https://huggingface.co/datasets/nameexhaustion/polars-docs/tree/main/hive_dates/](https://huggingface.co/datasets/nameexhaustion/polars-docs/tree/main/hive_dates/)

#### IPC

{{code_block('user-guide/io/hugging-face','scan_ipc',['scan_ipc'])}}

```python exec="on" result="text" session="user-guide/io/hugging-face"
--8<-- "python/user-guide/io/hugging-face.py:scan_ipc_repr"
```

See this file at [https://huggingface.co/spaces/nameexhaustion/polars-docs/blob/main/orders.feather](https://huggingface.co/spaces/nameexhaustion/polars-docs/blob/main/orders.feather)
