import shutil
from pathlib import Path

CWD = Path(__file__).parent


def substitute_template(s: str, rt: str) -> str:
    """Apply the runtime template substitutions."""
    s = s.replace("{{%RT_SUFFIX%}}", rt)
    return s


def generate_filetree(base_dir: Path, template_dirname: str, rt: str) -> None:
    """Copy template into a new runtime folder."""
    shutil.rmtree(base_dir, ignore_errors=True)
    shutil.copytree(CWD / template_dirname, base_dir)
    shutil.copyfile(
        CWD / ".." / ".." / "rust-toolchain.toml",
        base_dir / "rust-toolchain.toml",
    )
    shutil.move(base_dir / "_polars_runtime_mod", base_dir / f"_polars_runtime_{rt}")

    # Rename Cargo.template.toml to Cargo.toml. This rename is done to avoid
    # cargo from picking up the template file as a real Cargo.toml, see #25391.
    shutil.move(base_dir / "Cargo.template.toml", base_dir / "Cargo.toml")

    cargo_toml = (base_dir / "Cargo.toml").read_text()
    (base_dir / "Cargo.toml").write_text(substitute_template(cargo_toml, rt))

    pyproject_toml = (base_dir / "pyproject.toml").read_text()
    (base_dir / "pyproject.toml").write_text(substitute_template(pyproject_toml, rt))


if __name__ == "__main__":
    for rt in ["32", "64", "compat"]:
        base_dir = CWD / f"polars-runtime-{rt}"
        generate_filetree(base_dir, "template", rt)

        # Free-threaded variant: exactly same layout but with different Rust package
        # name (but identical Python package name to let PyPI/pip/uv handle it in the
        # same way), no ABI (free-threaded Python does not support it), and bumped
        # Python requirement (>=3.14).
        base_dir = CWD / f"polars-runtime-{rt}-ft"
        generate_filetree(base_dir, "template-ft", rt)
