A major point of `polars-on-premises` is the system requirements regarding the Python version and
Python dependencies, which need to be identical on the client, the scheduler, and all workers. The
easiest method to achieve this is having a system-wide Python environment and globally installed
packages.

The minimal requirement for running `polars-on-premises` is the `polars` package. Each release of
`polars-on-premises` is pinned to a single `polars` release, which can be found in the release
announcement.

## System-wide installation

```shell
$ uv pip install --break-system-packages -r requirements.txt polars[cloudpickle]==PINNED_VERSION
$ polars-on-premises service --config-path /etc/polars-cloud/config.toml
```

## Virtual Environment

It's also possible to run `polars-on-premises` using a virtual environment.

```shell
$ uv venv .venv
$ uv pip install -r requirements.txt polars[cloudpickle]==PINNED_VERSION
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
$ polars-on-premises service --config-path /etc/polars-cloud/config.toml
```
