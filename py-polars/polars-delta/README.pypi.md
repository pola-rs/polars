# polars-delta

Native Delta Lake scan planning plugin for compatible Polars builds. The plugin resolves
Delta snapshots, checkpoints, schemas and file metadata; Polars performs data reads and
applies predicates and deletion vectors.

The Python import module is `polars_delta`. It is normally used through `polars.scan_delta`,
not called directly. Compatibility is negotiated using versioned plugin contracts.

This development version requires a matching Polars build with the Delta planning contract.
