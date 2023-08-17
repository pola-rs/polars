import re
import subprocess
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
        return

    cmd = (
        f"ps -aux | grep 'debugpy/launcher/../../debugpy' "
        f"| grep '{__file__}' | awk '{{print $2}}'"
    )

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    output, _ = p.communicate()
    p_status = p.wait()

    # parse the output: the first should be the actual pid (the second is the Popened
    # subprocess)
    if p_status != 0 or output is None:
        raise RuntimeError("Cannot get pID of current debugging session.")
    output = output.decode()
    if (out := re.match("^(\\d+)", output)) is None:
        raise RuntimeError("Cannot get pID of current debugging session.")

    pID = int(out.group(0))

    # print to the console to allow the "Rust LLDB" routine to pick up on the signal
    print(f"pID = {pID}")

    # give the LLDB time to connect. We may have to play with this setting.
    time.sleep(1)

    # run the originally requested file
    # we updated sys.argv so that when exec() is called, it's populated with
    # the requested script name in sys.argv[0] and the remaining args after
    sys.argv = sys.argv[1:]
    exec(Path.open(sys.argv[0]).read())


if __name__ == "__main__":
    launch_debugging()
