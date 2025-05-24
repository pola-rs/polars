//! A tool for working with DSL schema.
//!
//! Usage:
//! `dsl-check [generate|update-hash|check-hash] [PATH]`
//! - `generate` the DSL schema as a JSON file in the current directory
//! - `update-hash` the schema hash stored in the schema hash file,
//! - `check-hash` that the schema hash in the file matches the code.
//!
//! The generated schema is affected by active features. To use a complete schema, first build
//! the whole workspace with all features:
//! ```sh
//! cargo build --all-features
//! ./target/debug/dsl-check update-hash
//! ./target/debug/dsl-check check-hash
//! ```

fn main() {
    #[cfg(not(feature = "dsl-schema"))]
    panic!("this tool requires the `dsl-schema` feature");

    #[cfg(feature = "dsl-schema")]
    {
        impls::run();
    }
}

#[cfg(feature = "dsl-schema")]
mod impls {
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    use polars_plan::dsl::DslPlan;
    use sha2::Digest;

    const DEFAULT_HASH_PATH: &str = "crates/polars-plan/dsl-schema.sha256";

    pub fn run() {
        let mut args = std::env::args();

        let _ = args.next();
        let cmd = args
            .next()
            .expect("missing command [generate, update-hash, check-hash]");
        let path = args.next();

        if let Some(unknown) = args.next() {
            panic!("unknown argument: `{unknown}`");
        }

        match cmd.as_str() {
            "generate" => {
                generate(path.unwrap_or("./dsl-schema.json".to_owned()));
            },
            "update-hash" => {
                update_hash(path.unwrap_or(DEFAULT_HASH_PATH.to_owned()));
            },
            "check-hash" => {
                check_hash(path.unwrap_or(DEFAULT_HASH_PATH.to_owned()));
            },
            unknown => {
                panic!("unknown command: `{unknown}`");
            },
        }
    }

    /// Serializes the current DSL schema into a file at the given path.
    ///
    /// Any existing file at the path is overwritten.
    fn generate(path: impl AsRef<Path>) {
        let schema = DslPlan::dsl_schema();

        let mut file = File::create(path).expect("failed to open the schema file for writing");
        serde_json::to_writer_pretty(&mut file, &schema).expect("failed to serialize the schema");
        writeln!(&mut file).expect("failed to write the last newline");
        file.flush().expect("failed to flush the schema file");
    }

    /// Outputs the current DSL schema hash into a file at the given path.
    ///
    /// Any existing file at the path is overwritten.
    fn update_hash(path: impl AsRef<Path>) {
        std::fs::write(path, current_hash()).expect("failed to write the hash into the file");
        eprintln!("the DSL schema hash file was updated");
    }

    /// Checks that the current schema hash matches the schema in the file.
    fn check_hash(path: impl AsRef<Path>) {
        let file_hash =
            std::fs::read_to_string(path).expect("faled to read the hash from the file");
        if file_hash != current_hash() {
            eprintln!(
                "the schema hash is not up to date, please run `make update-dsl-schema-hash`"
            );
            std::process::exit(1);
        }
        eprintln!("the DSL schema is up to date");
    }

    fn current_hash() -> String {
        let schema = DslPlan::dsl_schema();

        let mut digest = sha2::Sha256::new();
        serde_json::to_writer(&mut digest, &schema).expect("failed to serialize the schema");

        let hash = digest.finalize();

        format!("{hash:064x}")
    }
}
