import os
import re
import sys
import time
from pathlib import Path


def launch_debugging() -> None:
    """
    Debug Rust files via Python.

    Determine the pID for the current debugging session, attach the Rust LLDB launcher,
    and execute the originally-requested script.
    """
    if len(sys.argv) == 1:
        raise RuntimeError(
            "launch.py is not meant to be executed directly; please use the `Python: "
            "Debug Rust` debugging configuration to run a python script that uses the "
            "polars library."
        )

    # get the current process ID
    pID = os.getpid()

    # print to the console to allow the "Rust LLDB" routine to pick up on the signal
    launch_file = Path(__file__).parents[2] / ".vscode/launch.json"
    if not launch_file.exists():
        raise RuntimeError(f"Cannot locate {launch_file}")
    with launch_file.open("r") as f:
        launch_info = f.read()

    # overwrite the pid found in launch.config with the pid for the current process
    # match initial the "Rust LLDB" definition with the pid immediately after
    pattern = re.compile('("Rust LLDB",\\s*"pid":\\s*")\\d+(")')
    found = pattern.search(launch_info)
    if not found:
        raise RuntimeError(
            "Cannot locate pid definition in launch.json for Rust LLDB configuration. "
            "Please follow the instructions in CONTRIBUTING.md for creating the "
            "launch configuration."
        )

    launch_info_with_new_pid = pattern.sub(rf"\g<1>{pID}\g<2>", launch_info)
    with launch_file.open("w") as f:
        f.write(launch_info_with_new_pid)

    # print pID to the debug console. This auto-triggers the Rust LLDB configurations.
    print(f"pID = {pID}")

    # give the LLDB time to connect. We may have to play with this setting.
    time.sleep(1)

    # run the originally requested file
    # update sys.argv so that when exec() is called, it's populated with the requested
    # script name in sys.argv[0], and the remaining args after
    sys.argv.pop(0)
    with Path(sys.argv[0]).open() as fh:
        script_contents = fh.read()

    # path to the script to be executed
    fh = Path(sys.argv[0])
    exec(compile(script_contents, fh, mode="exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    launch_debugging()
