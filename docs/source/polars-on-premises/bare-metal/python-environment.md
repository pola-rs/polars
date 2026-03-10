## Virtual Environment

The simplest method is using the Python wheel. This ensures that your virtual environment is
properly set up to run Polars On-Premises. Note that when using Python UDFs, the server must have
the same Python packages installed as the client.

### Python wheel

To install the Python wheel, use our private PyPi index with your license key.

```shell
$ export LICENSE_KEY=$(cat license.json)
$ uv auth login https://get.onprem.pola.rs/pypi/simple --token $LICENSE_KEY
$ uv venv && source .venv/bin/activate
$ uv pip install --index-url=https://get.onprem.pola.rs/pypi/simple polars-on-premises==0.1.0
```

`polars-on-premises` is then available within your virtual environment and ready to get going.

```shell
$ polars-on-premises --version
```

### Direct binary download

When using the binary download directly, you need to ensure that it can access both Python and
Polars. The easiest method to achieve this is having a system-wide Python environment and globally
installed packages. We recommend however setting up a virtual environment
([`uv`](https://docs.astral.sh/uv/) makes this very easy, including maintaining a given Python
version).

!!! info "Version pinning" Each release of `polars-on-premises` is pinned to a single `polars`
release, which can be found in the release announcement and in `polars-on-premises --version`
`shell export PINNED_VERSION=1.38.1 # for instance`

#### System-wide installation

```shell
$ uv pip install --break-system-packages -r requirements.txt polars[cloudpickle]==$PINNED_VERSION
$ ./polars-on-premises --version
```

#### Virtual Environment

```shell
$ uv venv .venv
$ source .venv/bin/activate
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
$ uv pip install -r requirements.txt polars[cloudpickle]==$PINNED_VERSION
$ ./polars-on-premises service --config-path /etc/polars-cloud/config.toml
```
