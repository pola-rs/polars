# IDE configuration

Using an integrated development environments (IDE) and configuring it properly will help you work on Polars more effectively.
This page contains some recommendations for configuring popular IDEs.

## Visual Studio Code

The following extensions are recommended.

### rust-analyzer

If you work on the Rust code at all, you will need the [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer) extension. This extension provides code completion for the Rust code.

For it to work well for the Polars code base, add the following settings to your `.vscode/settings.json`:

```json
{
    "rust-analyzer.cargo.features": "all",
}
```

### Ruff

The Ruff extension will help you conform to the formatting requirements of the Python code.
We use both the Ruff linter and formatter.

## PyCharm / RustRover / CLion

!!! info

    More information needed.
