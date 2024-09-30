# IDE configuration

Using an integrated development environments (IDE) and configuring it properly will help you work on Polars more effectively.
This page contains some recommendations for configuring popular IDEs.

## Visual Studio Code

Make sure to configure VSCode to use the virtual environment created by the Makefile.

### Extensions

The extensions below are recommended.

#### rust-analyzer

If you work on the Rust code at all, you will need the [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer) extension. This extension provides code completion for the Rust code.

For it to work well for the Polars code base, add the following settings to your `.vscode/settings.json`:

```json
{
  "rust-analyzer.cargo.features": "all",
  "rust-analyzer.cargo.targetDir": true
}
```

#### Ruff

The [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) extension will help you conform to the formatting requirements of the Python code.
We use both the Ruff linter and formatter.
It is recommended to configure the extension to use the Ruff installed in your environment.
This will make it use the correct Ruff version and configuration.

```json
{
  "ruff.importStrategy": "fromEnvironment"
}
```

#### CodeLLDB

The [CodeLLDB](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb) extension is useful for debugging Rust code.
You can also debug Rust code called from Python (see section below).

### Debugging

Due to the way that Python and Rust interoperate, debugging the Rust side of development from Python calls can be difficult.
This guide shows how to set up a debugging environment that makes debugging Rust code called from a Python script painless.

#### Preparation

Start by installing the CodeLLDB extension (see above).
Then add the following two configurations to your `launch.json` file.
This file is usually found in the `.vscode` folder of your project root.
See the [official VSCode documentation](https://code.visualstudio.com/docs/editor/debugging#_launch-configurations) for more information about the `launch.json` file.

<details><summary><code><b>launch.json</b></code></summary>

```json
{
  "configurations": [
    {
      "name": "Debug Rust/Python",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/py-polars/debug/launch.py",
      "args": [
        "${file}"
      ],
      "console": "internalConsole",
      "justMyCode": true,
      "serverReadyAction": {
        "pattern": "pID = ([0-9]+)",
        "action": "startDebugging",
        "name": "Rust LLDB"
      }
    },
    {
      "name": "Rust LLDB",
      "pid": "0",
      "type": "lldb",
      "request": "attach",
      "program": "${workspaceFolder}/py-polars/.venv/bin/python",
      "stopOnEntry": false,
      "sourceLanguages": [
        "rust"
      ],
      "presentation": {
        "hidden": true
      }
    }
  ]
}
```

</details>

!!! info

    On some systems, the LLDB debugger will not attach unless [ptrace protection](https://linux-audit.com/protect-ptrace-processes-kernel-yama-ptrace_scope) is disabled.
    To disable, run the following command:

    ```shell
    echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
    ```

#### Running the debugger

1. Create a Python script containing Polars code. Ensure that your virtual environment is activated.

2. Set breakpoints in any `.rs` or `.py` file.

3. In the `Run and Debug` panel on the left, select `Debug Rust/Python` from the drop-down menu on top and click the `Start Debugging` button.

At this point, your debugger should stop on breakpoints in any `.rs` file located within the codebase.

#### Details

The debugging feature runs via the specially-designed VSCode launch configuration shown above.
The initial Python debugger is launched using a special launch script located at `py-polars/debug/launch.py` and passes the name of the script to be debugged (the target script) as an input argument.
The launch script determines the process ID, writes this value into the `launch.json` configuration file, compiles the target script and runs it in the current environment.
At this point, a second (Rust) debugger is attached to the Python debugger.
The result is two simultaneous debuggers operating on the same running instance.
Breakpoints in the Python code will stop on the Python debugger and breakpoints in the Rust code will stop on the Rust debugger.

## JetBrains (PyCharm, RustRover, CLion)

!!! info

    More information needed.
