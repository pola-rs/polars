# polars

`polars` runs SQL on files with the Polars engine, without Python. It is unstable: its options and
output can still change. It replaces the earlier `polars-cli` (up to 0.9.0): there is no interactive
shell, and `-o` is now `--format`.

```shell
polars -c "SELECT city, avg(temp) FROM 'weather/*.parquet' GROUP BY city"
polars -f report.sql --format csv > report.csv
```

## What it reads

- **Formats:** Parquet, CSV, NDJSON and Arrow IPC, named by path (`FROM 'data.parquet'`) or read
  with options through `read_parquet`, `read_csv`, `read_ndjson` and `read_ipc`.
- **Locations:** local files and globs, S3 and S3-compatible stores, Azure, Google Cloud Storage,
  HTTP(S) and public Hugging Face repositories (`hf://`).
- **Credentials:** environment variables, `~/.aws/credentials`, Google application-default
  credentials, and the identity of the machine or container it runs on. SQL takes no
  `storage_options`.

## Coming later

These need Python Polars for now. They are planned for `polars` in later releases.

- Delta Lake and Iceberg tables, and catalogs. Until then, reading their files directly can return
  removed data.
- More credential sources, such as AWS named profiles and SSO, which Python Polars gets from
  `boto3`, `azure-identity` and `google-auth`.
- Excel and databases.

## CPU

The x86-64 release builds need AVX2, FMA and BMI2: Intel Core and Xeon since Haswell (2013), AMD
since Excavator (2015). Many Pentium, Celeron and Atom chips lack AVX2. Without it, `polars` stops
with `Illegal instruction`. In a virtual machine, give it the host's CPU type; defaults such as
Proxmox's `kvm64` and `x86-64-v2-AES` hide AVX2.
