# Contribution guideline examples

This subdirectory is intended to provide examples that guide contributors.

Naming conventions for variables:

```rust
let s: Series = ...
let ca: ChunkedArray = ...
let arr: ArrayRef = ...
let arr: PrimitiveArray = ...
let dtype: DataType = ...
let data_type: ArrowDataType = ...
```

# Visual Studio Code settings

In case you use Visual Studio Code, the following settings are recommended
for rust-analyzer to detect all code:

```json
{
    "rust-analyzer.cargo.features": "all",
    "rust-analyzer.linkedProjects": ["py-polars/Cargo.toml"],
}
```

You can place this in `.vscode/settings.json` in the repository root.
