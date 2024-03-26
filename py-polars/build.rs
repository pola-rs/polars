/// Build script using 'built' crate to generate build info.

fn main() {
    #[cfg(feature = "build_info")]
    {
        println!("cargo:rerun-if-changed=build.rs");
        extern crate built;
        use std::env;
        use std::path::Path;

        // We must specify the workspace root as the source
        let py_polars_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let src = Path::new(&py_polars_dir).parent().unwrap();

        let out_dir = &env::var("OUT_DIR").unwrap();
        let dst = Path::new(&out_dir).join("built.rs");

        built::write_built_file_with_opts(Some(src), &dst)
            .expect("Failed to acquire build-time information");
    }

    // Add a pythonXY alias for Windows, so we can use CPython APIs that aren't
    // part of pyo3 (the tracemalloc APIs, see src/memory.rs).
    #[cfg(all(target_os = "windows", debug_assertions))]
    {
        let interpreter_config = pyo3_build_config::get();
        println!(
            "cargo:rustc-link-lib=pythonXY:{lib_name}",
            lib_name = interpreter_config.lib_name.as_ref().unwrap(),
        );
    }
}
