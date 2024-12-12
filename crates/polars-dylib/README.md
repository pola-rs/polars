# Polars dynamic library

```toml
# Cargo.toml
[workspace.dependencies.polars]
package = "polars-dylib"
```

```toml
# .cargo/config.toml
[build]
rustflags = [
  "-C",
  "prefer-dynamic",
]
```
