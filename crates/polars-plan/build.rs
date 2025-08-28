use std::env;
use std::io::Write;
use std::path::Path;

use sha2::Digest;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let channel = version_check::Channel::read().unwrap();
    if channel.is_nightly() {
        println!("cargo:rustc-cfg=feature=\"nightly\"");
    }

    generate_schema_hash();
}

/// Generate a hash of the schema hashes file, to be embedded in the binary.
///
/// Used in `SchemaHash` (crates/polars-plan/src/dsl/plan.rs) for DSL compatibility check.
fn generate_schema_hash() {
    let hash_hexstr = {
        let mut digest = sha2::Sha256::new();
        digest
            .write_all(include_bytes!("dsl-schema-hashes.json"))
            .expect("failed to hash the schema hashes file");
        let hash = digest.finalize();

        format!("{hash:064x}")
    };

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dst = Path::new(&out_dir).join("dsl-schema.sha256");
    std::fs::write(dst, &hash_hexstr).expect("failed to write the schema hash file");

    println!("cargo:rerun-if-changed=dsl-schema-hashes.json");
}
