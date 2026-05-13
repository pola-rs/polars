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
build = toml.get("build", {})
env["PYO3_ENVIRONMENT_SIGNATURE"] = (
    f"cpython-{sys.version_info[0]}.{sys.version_info[1]}-64bit"
)
env["PYO3_PYTHON"] = str(python_path)

if os.environ.get("RUSTFLAGS"):
    build["rustflags"] = os.environ.get("RUSTFLAGS")
else:
    build.pop("rustflags", None)

if os.environ.get("CFLAGS"):
    env["CFLAGS"] = os.environ.get("CFLAGS")
else:
    env.pop("CFLAGS", None)

toml["env"] = env

# On linux, building with dev profile might fail at link time, because debug symbols
# exceed 4 GB. Unless "profile.dev.split-debuginfo" is set already, set it to "unpacked"
if sys.platform == "linux":
    profile = toml.get("profile", {})
    dev = profile.get("dev", {})

    if dev.get("split-debuginfo") is None:
        dev["split-debuginfo"] = "unpacked"
        profile["dev"] = dev
        toml["profile"] = profile


with config_toml.open("w") as f:
    tomlkit.dump(toml, f)
