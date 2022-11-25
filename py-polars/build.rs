/// Build script using 'built' crate to generate build info.

fn main() {
    #[cfg(feature = "build_info")]
    {
        println!("cargo:rerun-if-changed=build.rs");
        extern crate built;
        use std::env;
        use std::path::Path;

        let src = env::var("CARGO_MANIFEST_DIR").unwrap();
        let dst = Path::new(&env::var("OUT_DIR").unwrap()).join("built.rs");
        let mut opts = built::Options::default();

        opts.set_dependencies(true).set_compiler(true).set_env(true);

        built::write_built_file_with_opts(&opts, Path::new(&src), &dst)
            .expect("Failed to acquire build-time information");
    }
}
