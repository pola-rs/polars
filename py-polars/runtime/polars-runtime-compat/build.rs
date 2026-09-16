use std::path::PathBuf;

use rustflags::Flag;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(allocator, values(\"default\", \"mimalloc\"))");
    println!(
        "cargo:rustc-env=TARGET={}",
        std::env::var("TARGET").unwrap()
    );

    // Write out feature flags for runtime compatibility checks.
    // We don't use OUT_DIR for this because maturin only includes files in the
    // source directory. Since we generate a Python file and nothing the Rust
    // compiler expects this should be fine.
    let mut target_feats = String::new();
    for flag in rustflags::from_env() {
        if let Flag::Codegen {
            opt,
            value: Some(value),
        } = flag
        {
            if opt == "target-feature" {
                if !target_feats.is_empty() {
                    target_feats.push(',');
                }
                target_feats.push_str(&value);
            }
        }
    }

    // The Python source folder sits next to the pyproject.toml. Normally that is
    // this crate's directory, but when building from an sdist maturin puts the
    // pyproject.toml (and with it the Python source folder) at the sdist root,
    // with this crate nested underneath at py-polars/runtime/<name>. So walk up
    // until we find it.
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let runtime_folder = manifest_dir
        .ancestors()
        .take(4)
        .find_map(|dir| {
            std::fs::read_dir(dir)
                .ok()?
                .filter_map(|entry| entry.ok())
                .find(|entry| {
                    entry
                        .file_name()
                        .to_string_lossy()
                        .starts_with("_polars_runtime")
                        && entry.file_type().is_ok_and(|t| t.is_dir())
                })
        })
        .unwrap_or_else(|| {
            panic!("could not find _polars_runtime* source folder at or above {manifest_dir:?}")
        });

    std::fs::write(
        runtime_folder.path().join("build_feature_flags.py"),
        format!("BUILD_FEATURE_FLAGS = \"{target_feats}\"\n"),
    )
    .unwrap();
}
