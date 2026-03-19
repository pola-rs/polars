import os
import re
import sys
import time
from pathlib import Path

"""
The following parameter determines the sleep time of the Python process after a signal
is sent that attaches the Rust LLDB debugger. If the Rust LLDB debugger attaches to the
current session too late, it might miss any set breakpoints. If this happens
consistently, it is recommended to increase this value.
"""
LLDB_DEBUG_WAIT_TIME_SECONDS = 1


def launch_debugging() -> None:
    """
    Debug Rust files via Python.

    Determine the pID for the current debugging session, attach the Rust LLDB launcher,
    and execute the originally-requested script.
    """
    if len(sys.argv) == 1:
        msg = (
            "launch.py is not meant to be executed directly; please use the `Python: "
            "Debug Rust` debugging configuration to run a python script that uses the "
            "polars library."
        )
        raise RuntimeError(msg)

    # Get the current process ID.
    pID = os.getpid()

    # Print to the debug console to allow VSCode to pick up on the signal and start the
    # Rust LLDB configuration automatically.
    launch_file = Path(__file__).parents[2] / ".vscode/launch.json"
    if not launch_file.exists():
        msg = f"Cannot locate {launch_file}"
        raise RuntimeError(msg)
    with launch_file.open("r") as f:
        launch_info = f.read()

    # Overwrite the pid found in launch.json with the pid for the current process.
    # Match the initial "Rust LLDB" definition with the pid defined immediately after.
    pattern = re.compile('("Rust LLDB",\\s*"pid":\\s*")\\d+(")')
    found = pattern.search(launch_info)
    if not found:
        msg = (
            "Cannot locate pid definition in launch.json for Rust LLDB configuration. "
            "Please follow the instructions in the debugging section of the "
            "contributing guide (https://docs.pola.rs/development/contributing/ide/#debugging) "
            "for creating the launch configuration."
        )
        raise RuntimeError(msg)

    launch_info_with_new_pid = pattern.sub(rf"\g<1>{pID}\g<2>", launch_info)
    with launch_file.open("w") as f:
        f.write(launch_info_with_new_pid)

    # Print pID to the debug console. This auto-triggers the Rust LLDB configurations.
    print(f"pID = {pID}")

    # Give the LLDB time to connect. Depending on how long it takes for your LLDB
    # debugging session to initialize, you may have to adjust this setting.
    time.sleep(LLDB_DEBUG_WAIT_TIME_SECONDS)

    # Update sys.argv so that when exec() is called, the first argument is the script
    # name itself, and the remaining are the input arguments.
    sys.argv.pop(0)
    with Path(sys.argv[0]).open() as fh:
        script_contents = fh.read()

    # Run the originally requested file by reading in the script, compiling, and
    # executing the code.
    file_to_execute = Path(sys.argv[0])
    exec(
        compile(script_contents, file_to_execute, mode="exec"), {"__name__": "__main__"}
    )


if __name__ == "__main__":
    launch_debugging()
