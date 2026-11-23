fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Embedded as `__build_commit__` of the Python module (see `python/mod.rs`).
    println!("cargo:rerun-if-env-changed=POLARS_DATABASE_BUILD_COMMIT");
}
