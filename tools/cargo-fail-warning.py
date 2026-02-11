# A wrapper for cargo which fails on warnings.
# This is better than -D warnings as it will continue the build rather than fail early.

import subprocess
import json
import sys

subcommand = sys.argv[1]

proc = subprocess.Popen(
    ["cargo", subcommand, "--message-format=json-diagnostic-rendered-ansi"]
    + sys.argv[2:],
    stdout=subprocess.PIPE,
)

found_warning = False
for line in iter(proc.stdout.readline, b""):
    msg = json.loads(line)
    if msg["reason"] in (
        "compiler-artifact",
        "build-script-executed",
        "build-finished",
    ):
        continue

    if msg["reason"] == "compiler-message":
        print(msg["message"]["rendered"], file=sys.stderr)
        if msg["message"]["level"] == "warning":
            found_warning = True
        continue

    raise RuntimeError("unknown message reason:\n", json.dumps(msg, indent=4))

# Respect original exit code if non-zero, otherwise emit if warning was found.
exit = proc.wait()
sys.exit(exit if exit != 0 else int(found_warning))
