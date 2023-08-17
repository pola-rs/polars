# Debugging Rust/Python code from the VSCode Debugger

The file `launch.py` found in this folder is intended for use in debugging in VSCode. It provides an easy mechanism to run a
python script containing Polars code and, in the process, halt on any breakpoints in either Python or Rust. It accomplishes
this by first initiating a Python debugging session, then attaching the Rust LLDB debugger to the same process. The result is
two simultaneous debuggers that can break on either Python or Rust code.

The `launch.py` file found in this directory is intended **only** to be run via the `Python: Debug Rust` VSCode debugging
configuration. It is **not** intended to be run via the command line.

## Setup


### 1. Install The [CodeLLDB](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb) extension.

You can install
from within a VS Code terminal with the following:

```shell
code --install-extension vadimcn.vscode-lldb augustocdias.tasks-shell-input
```

### 2. Add debug launch configurations.

Copy the following two configurations to your `launch.json` file. This file is usually found in the `.vscode` subfolder of your
root directory. See [Debugging in VSCode](https://code.visualstudio.com/docs/editor/debugging#_launch-configurations) for more
information about the `launch.json` file.

<details><summary><code><b>launch.json</b></code></summary>

```json
{
    "configurations": [
        {
            "name": "Python: Debug Rust",
            "type": "python",
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
<br>

### Debugging a script

1. Create a python script containing polars code. Ensure that your polars virtual environment is activated.

2. Set breakpoingt in any `.rs` or `.py` file.

3. In the `Run and Debug` panel on the left, select `Python: Debug Rust` from the drop-down menu on top and click
the `Start Debugging` button.

At this point, your debugger should stop on breakpoints in any `.rs` file located within the codebase. To quickly
re-start the debugger in the future, use the standard `F5` keyboard shortcut to re-launch the `Python: Debug Rust`
debugging configuration.

#### Details

The debugging feature runs via the specially-designed VS Code launch configuration shown above. The initial python debugger
is launched, using a special launch script located at `/py-polars/debug/launch.py`, and passes the name of the script to be
debugged (the target script) as an input argument. The launch script determines the process ID, writes this value into
the launch.json configuration file, compiles the target script and runs it in the current environment. At this point, a
second (Rust) debugger is attached to the Python debugger. The result is two simultaneous debuggers operating on the same
running instance. Breakpoints in the Python code will stop on the Python debugger and breakpoints in the Rust code will stop
on the Rust debugger.
