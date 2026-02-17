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

    let runtime_folder = std::fs::read_dir(".")
        .unwrap()
        .filter_map(|entry| entry.ok())
        .find(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("_polars_runtime")
        })
        .unwrap();

    std::fs::write(
        runtime_folder.path().join("build_feature_flags.py"),
        format!("BUILD_FEATURE_FLAGS = \"{target_feats}\"\n"),
    )
    .unwrap();
}
