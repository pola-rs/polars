# polars-delta

Native Delta Lake scan planning plugin for compatible Polars builds. The plugin uses Delta Kernel
for Rust to resolve snapshots and replay the transaction log. Polars reads the selected data files
and applies predicates and deletion vectors.

The Python import module is `polars_delta`. It is normally used through `polars.scan_delta`, not
called directly. Compatibility is negotiated using versioned plugin contracts.

This development version requires a matching Polars build with the Delta planning contract.
