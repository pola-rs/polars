# Plugins and custom libraries

Polars Cloud supports extending functionality through plugins and user-defined functions (UDFs).
This guide covers how to use these extensions in your workflows.

## Expression Plugins

[Expression plugins](https://docs.pola.rs/user-guide/plugins/) are the preferred way to extend
Polars functionality in cloud environments. They work seamlessly with Polars Cloud's remote
execution.

Plugins are compiled Rust functions that integrate directly into the Polars expression engine,
allowing you to add custom functionality that behaves like native Polars operations. They're
distributed as Python packages that register new expression methods.

### Setting Up Your Environment

Start by creating a `requirements.txt` file with your dependencies.

```python
# requirements.txt

numpy==2.2.4
polars-xdt==0.16.8
```

After that, include the file in the compute context. In the compute context, you can specify a
`requirements.txt` file with additional packages to install on your compute instance.

{{code_block('polars-cloud/plugins','set-context',['ComputeContext'])}}

### Using Plugins in Remote Workflows

In this example query we use the [polars-xdt](https://github.com/pola-rs/polars-xdt) plugin. This
plugin offers extra datetime-related functionality which isn't in-scope for the main Polars library.

Once installed, plugins extend the Polars expression API with new namespaces and methods. When
importing the plugin, we can use its functionality just as on a local machine:

{{code_block('polars-cloud/plugins','run-plugin',[])}}

```
shape: (3, 3)
┌─────────────────────┬──────────────────┬─────────────────────────┐
│ local_dt            ┆ timezone         ┆ date                    │
│ ---                 ┆ ---              ┆ ---                     │
│ datetime[μs]        ┆ str              ┆ datetime[μs, UTC]       │
╞═════════════════════╪══════════════════╪═════════════════════════╡
│ 2020-10-10 01:00:00 ┆ Europe/London    ┆ 2020-10-10 00:00:00 UTC │
│ 2020-10-10 02:00:00 ┆ Africa/Kigali    ┆ 2020-10-10 00:00:00 UTC │
│ 2020-10-09 20:00:00 ┆ America/New_York ┆ 2020-10-10 00:00:00 UTC │
└─────────────────────┴──────────────────┴─────────────────────────┘
```

Plugins generally offer better performance than UDFs since they run compiled code. However, UDFs
provide more flexibility for custom logic.

### Using Other Libraries in Remote Execution

The process of using a
[user-defined function](https://docs.pola.rs/user-guide/expressions/user-defined-python-functions/)
in your workflow is the same as with plugins. Ensure your dependencies are included in the
requirements file you pass in the compute context. Your custom Python functions will execute on the
remote compute instances:

{{code_block('polars-cloud/plugins','run-udf',[])}}

```
shape: (4, 1)
┌──────────┐
│ values   │
│ ---      │
│ f64      │
╞══════════╡
│ 2.302585 │
│ 1.94591  │
│ 0.0      │
│ 3.135494 │
└──────────┘
```
