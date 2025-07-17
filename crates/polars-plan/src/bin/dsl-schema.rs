//! A tool for working with DSL schema.
//!
//! Usage:
//! `dsl-schema [generate|update|check] [PATH]`
//! - `generate` the DSL schema as a full JSON file in the current directory
//! - `update` the schema stored in the schema file (stored as JSON Lines),
//! - `check` that the schema in the file (stored as JSON Lines) matches the code.
//!
//! The generated schema is affected by active features. To use a complete schema, first build
//! the whole workspace with all features:
//! ```sh
//! cargo build --all-features
//! ./target/debug/dsl-schema update
//! ./target/debug/dsl-schema check
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
    use std::io::Write as _;
    use std::path::Path;

    use polars_core::prelude::{InitHashMaps, PlHashMap};
    use polars_plan::dsl::DslPlan;
    use schemars::schema::SchemaObject;

    const DEFAULT_SCHEMA_PATH: &str = "crates/polars-plan/dsl-schema.jsonl";

    pub fn run() {
        let mut args = std::env::args();

        let _ = args.next();
        let cmd = args
            .next()
            .expect("missing command [generate, update, check]");
        let path = args.next();

        if let Some(unknown) = args.next() {
            panic!("unknown argument: `{unknown}`");
        }

        match cmd.as_str() {
            "generate" => {
                generate(path.unwrap_or("./dsl-schema.json".to_owned()));
            },
            "update" => {
                update(path.unwrap_or(DEFAULT_SCHEMA_PATH.to_owned()));
            },
            "check" => {
                check(path.unwrap_or(DEFAULT_SCHEMA_PATH.to_owned()));
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

    /// Outputs the current DSL schema into a file at the given path.
    ///
    /// Any existing file at the path is overwritten.
    fn update(path: impl AsRef<Path>) {
        std::fs::write(path, current_schema_json_lines())
            .expect("failed to write the schema into the file");
        eprintln!("the DSL schema file was updated");
    }

    /// Checks that the current schema matches the schema in the file.
    fn check(path: impl AsRef<Path>) {
        let file_schema =
            std::fs::read_to_string(path).expect("faled to read the schema from the file");
        if file_schema != current_schema_json_lines() {
            eprintln!(
                "the schema is not up to date, please update the DSL_VERSION in `polars-plan/src/dsl/plan.rs` if needed and run `make update-dsl-schema`"
            );
            std::process::exit(1);
        }
        eprintln!("the DSL schema is up to date");
    }

    /// Returns the schema as JSON Lines, one object per line, each object has a single field
    /// named after the data type, with the schema as the value.
    fn current_schema_json_lines() -> String {
        let mut lines = Vec::new();

        {
            let mut map = PlHashMap::with_capacity(1);
            let mut push = |name: String, schema: SchemaObject| {
                map.insert(name, schema);
                lines.push(serde_json::to_string(&map).expect("failed to serialize the schema"));
                map.clear();
            };

            // Serialize he schemas as an object per line
            let mut schema = DslPlan::dsl_schema();

            // The first line is the top level enum
            push(String::from("DslPlan"), schema.schema);

            // The rest is sorted by name
            schema.definitions.sort_keys();

            for (name, def) in schema.definitions {
                push(name, def.into_object());
            }
        }

        let mut out = lines.join("\n");
        out.push('\n');

        out
    }
}
