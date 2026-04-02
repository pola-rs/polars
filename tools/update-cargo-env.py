import sys
import os
import tomlkit
from pathlib import Path

python_path = Path(sys.executable).absolute()
cargo_dir = Path(__file__).absolute().parent.parent / ".cargo"
cargo_dir.mkdir(exist_ok=True)
config_toml = cargo_dir / "config.toml"

try:
    with config_toml.open("rb") as f:
        toml = tomlkit.load(f)
except FileNotFoundError:
    toml = {}

env = toml.get("env", {})
env["PYO3_ENVIRONMENT_SIGNATURE"] = (
    f"cpython-{sys.version_info[0]}.{sys.version_info[1]}-64bit"
)
env["PYO3_PYTHON"] = str(python_path)

if os.environ.get("RUSTFLAGS"):
    env["RUSTFLAGS"] = os.environ.get("RUSTFLAGS")
else:
    env.pop("RUSTFLAGS", None)

if os.environ.get("CFLAGS"):
    env["CFLAGS"] = os.environ.get("CFLAGS")
else:
    env.pop("CFLAGS", None)

toml["env"] = env

with config_toml.open("w") as f:
    tomlkit.dump(toml, f)
