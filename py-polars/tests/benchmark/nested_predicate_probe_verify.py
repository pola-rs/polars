"""Save the temporary runtime and verify modes; run on the build remote."""

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

out = Path("target/nested-followup")
out.mkdir(parents=True, exist_ok=True)
runtime = Path(importlib.import_module("_polars_runtime_32._polars_runtime").__file__)
saved = out / "probe.so"
shutil.copy2(runtime, saved)
metadata = {
    "runtime": str(runtime),
    "bytes": saved.stat().st_size,
    "sha256": hashlib.file_digest(saved.open("rb"), "sha256").hexdigest(),
}
(out / "binary.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata), flush=True)
for mode, mask in [("auto", "auto"), ("root", "off"), ("leaf", "off"), ("leaf", "on")]:
    env = dict(
        os.environ,
        POLARS_MAX_THREADS="5",
        POLARS_NESTED_PROBE_MODE=mode,
        POLARS_NESTED_PROBE_MASK=mask,
    )
    env.pop("POLARS_NESTED_PROBE_SKIP_ROOT", None)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "py-polars/tests/unit/io/test_nested_predicate_pushdown.py",
        "py-polars/tests/unit/io/test_struct_field_pruning.py",
        "py-polars/tests/unit/io/test_struct_field_pruning_lists.py",
        "-x",
    ]
    if mode == "root":
        command.extend(
            ["-k", "not test_nested_leaf_rejects_rows_before_decoding_sibling"]
        )
    log = out / f"verify-{mode}-{mask}-final.log"
    with log.open("w") as stream:
        result = subprocess.run(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    print(
        mode,
        mask,
        "exit",
        result.returncode,
        log.read_text().splitlines()[-1],
        flush=True,
    )
    assert result.returncode == 0
