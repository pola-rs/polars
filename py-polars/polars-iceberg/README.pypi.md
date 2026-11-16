# polars-iceberg

This package is a dependency of [Polars](https://pypi.org/project/polars/) and is not meant to be
used directly. It contains the native Iceberg scan planning plugin that `polars.scan_iceberg`
uses.

To install it, install Polars with the `iceberg` extra:

```bash
pip install 'polars[iceberg]'
```

See the [Polars documentation](https://docs.pola.rs/) and the
[Polars repository](https://github.com/pola-rs/polars) for usage and issues.
