//! A tool for working with DSL schema.
//!
//! Usage:
//! `dsl-schema [generate|update-hashes|check-hashes] [PATH]`
//! - `generate` the DSL schema as a full JSON file in the current directory
//! - `update-hashes` stored in the schema hashes file,
//! - `check-hashes` in the schema hashes file against the hashes from the code.
//!
//! The generated schema is affected by active features. To use a complete schema, first build
//! the whole workspace with all features:
//! ```sh
//! cargo build --all-features
//! ./target/debug/dsl-schema update-hashes
//! ./target/debug/dsl-schema check-hashes
//!
//! The tool has the code schema built-in. After code changes, you need to run
//! `cargo build --all-features` again.
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
    use schemars::schema::SchemaObject;
    use sha2::Digest;

    const DEFAULT_HASHES_PATH: &str = "crates/polars-plan/dsl-schema-hashes.json";

    pub fn run() {
        let mut args = std::env::args();

        let _ = args.next();
        let cmd = args
            .next()
            .expect("missing command [generate, update-hashes, check-hashes]");
        let path = args.next();

        if let Some(unknown) = args.next() {
            panic!("unknown argument: `{unknown}`");
        }

        match cmd.as_str() {
            "generate" => {
                generate(path.unwrap_or("./dsl-schema.json".to_owned()));
            },
            "update-hashes" => {
                update_hashes(path.unwrap_or(DEFAULT_HASHES_PATH.to_owned()));
            },
            "check-hashes" => {
                check_hashes(path.unwrap_or(DEFAULT_HASHES_PATH.to_owned()));
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

    /// Outputs the current DSL schema hashes into a file at the given path.
    ///
    /// Any existing file at the path is overwritten.
    fn update_hashes(path: impl AsRef<Path>) {
        std::fs::write(path, current_schema_hashes())
            .expect("failed to write the schema into the file");
        eprintln!("the DSL schema file was updated");
    }

    /// Checks that the current schema hashes match the schema hashes in the file.
    fn check_hashes(path: impl AsRef<Path>) {
        let file_hashes =
            std::fs::read_to_string(path).expect("failed to read the schema hashes from the file");
        if file_hashes != current_schema_hashes() {
            eprintln!("the schema hashes are not up to date, run `make update-dsl-schema-hashes`");
            std::process::exit(1);
        }
        eprintln!("the DSL schema hashes are up to date");
    }

    /// Returns the schema hashes as a serialized JSON object.
    /// Each field is named after a data type, with its schema hash as the value.
    fn current_schema_hashes() -> String {
        let schema = DslPlan::dsl_schema();

        let mut hashes = serde_json::Map::new();

        // Insert the top level enum schema
        hashes.insert(String::from("DslPlan"), schema_hash(&schema.schema).into());

        // Insert the subschemas
        for (name, def) in schema.definitions {
            hashes.insert(name, schema_hash(&def.into_object()).into());
        }

        hashes.sort_keys();

        serde_json::to_string_pretty(&hashes).expect("failed to serialize schema hashes file")
    }

    fn schema_hash(schema: &SchemaObject) -> String {
        let mut digest = sha2::Sha256::new();
        serde_json::to_writer(&mut digest, schema).expect("failed to serialize the schema");
        let hash = digest.finalize();
        format!("{hash:064x}")
    }
}
