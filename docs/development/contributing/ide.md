# IDE configuration

Using an integrated development environments (IDE) and configuring it properly will help you work on Polars more effectively.
This page contains some recommendations for configuring popular IDEs.

## Visual Studio Code

### Python environment

Make sure to configure VSCode to use the virtual environment created by the Makefile.
VSCode will prompt you to do this automatically once the Makefile creates the `.venv` folder.
You can also run the `Python: Select Interpreter` command to manually configure the workspace to use the local `.venv` interpreter.

### Line lengths

We use 88 characters for Python code and 100 characters for Rust code. If you want VSCode to show you rulers at those points, add the following settings to your `.vscode/settings.json`:

```json
{
    "[python]": {
        "editor.rulers": [
            88
        ],
    },
    "[rust]": {
        "editor.rulers": [
            100
        ],
    },
}
```

### Extensions

We recommend using the following VSCode extensions to aid your development experience.

#### rust-analyzer

If you work on the Rust code at all, you will need the [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer) extension. This extension provides code completion for the Rust code.

For it to work well for the Polars code base, add the following settings to your `.vscode/settings.json`:

```json
{
    "rust-analyzer.cargo.features": "all",
}
```

#### Ruff

We use the Ruff linter and formatter on our Python code. The [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) extension will help you conform to the formatting requirements of the Python code.

If you want to configure Ruff to format automatically on save, add the following settings to your `.vscode/settings.json`:

```json
{
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff",
    }
}
```

You should also configure the extension to use the Ruff installed in your virtual environment.
This will make it use the correct Ruff version and configuration.

```json
{
    "ruff.importStrategy": "fromEnvironment",
}
```

## PyCharm / RustRover / CLion

!!! info

    More information needed.
