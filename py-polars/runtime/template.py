import shutil
from pathlib import Path

this_dir = Path(__file__).parent


def template(s: str, rt: str) -> str:
    """Apply the runtime template substitutions."""
    s = s.replace("{{%RT_SUFFIX%}}", rt)
    return s


if __name__ == "__main__":
    for rt in ["32", "64", "compat"]:
        basedir = this_dir / Path("polars-runtime-" + rt)
        shutil.rmtree(basedir, ignore_errors=True)
        shutil.copytree(this_dir / "template", basedir)
        shutil.copyfile(this_dir / ".." / ".." / "rust-toolchain.toml", basedir / "rust-toolchain.toml")
        shutil.move(
            basedir / "_polars_runtime_mod", basedir / ("_polars_runtime_" + rt)
        )

        with (basedir / "Cargo.toml").open() as f:
            cargo_toml = f.read()
        with (basedir / "pyproject.toml").open() as f:
            pyproject_toml = f.read()

        with (basedir / "Cargo.toml").open("w") as f:
            f.write(template(cargo_toml, rt))
        with (basedir / "pyproject.toml").open("w") as f:
            f.write(template(pyproject_toml, rt))
