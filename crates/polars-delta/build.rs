fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Embedded as `__build_commit__` of the Python module (see `src/python.rs`).
    println!("cargo:rerun-if-env-changed=POLARS_DELTA_BUILD_COMMIT");
}
