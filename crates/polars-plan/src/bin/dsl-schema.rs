//! A tool for working with DSL schema.
//!
//! This tool can
//! - update the schema stored in the schema file,
//! - check that the schema in the file matches the code.
//!
//! The generated schema is affected by active features. To use a complete schema, first build
//! the whole workspace with all features:
//! ```sh
//! cargo build --all-features
//! ./target/debug/dsl-check update
//! ./target/debug/dsl-check check
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

    use polars_plan::dsl::DslPlan;
    use schemars::schema::RootSchema;

    const DEFAULT_PATH: &str = "crates/polars-plan/dsl-schema.json";

    pub fn run() {
        let mut args = std::env::args();

        let _ = args.next();
        let cmd = args.next().expect("missing command [update, check]");
        let path_arg = args.next().unwrap_or(DEFAULT_PATH.to_owned());

        if let Some(unknown) = args.next() {
            panic!("unknown argument: `{unknown}`");
        }

        let path = Path::new(&path_arg);
        match cmd.as_str() {
            "update" => {
                update(path);
            },
            "check" => {
                check(path);
            },
            unknown => {
                panic!("unknown command: `{unknown}`");
            },
        }
    }

    /// Generates serializes the current DSL schema into a file at the given path.
    ///
    /// Any existing file at the path is overwritten.
    fn update(path: &Path) {
        let schema = DslPlan::dsl_schema();

        let mut file = File::create(path).expect("failed to create a writable schema file");
        serde_json::to_writer_pretty(&mut file, &schema).expect("failed to serialize the schema");
        writeln!(&mut file).expect("failed to write the last newline");
        file.flush().expect("failed to flush the schema file");

        eprintln!("the DSL schema file was updated");
    }

    /// Checks that the current schema matches the schema in the file.
    fn check(path: &Path) {
        let schema: RootSchema = {
            // Do a serialization round trip, because `schemars` has a quirk where serializing some
            // fields with a value of `Some(Default::default())` gets deserialized as `None`. Both
            // values mean the same thing, but `PartialEq` treats them as not equal.
            // The serialization round trip normalizes all these fields to `None`, so that
            // `PartialEq` works as expected.
            let json = serde_json::to_string(&DslPlan::dsl_schema())
                .expect("failed to serialize the current schema");
            serde_json::from_str(&json).expect("failed to deserialize the current schema")
        };

        let mut file = File::open(path).expect("failed to open the schema file");
        let schema_file: RootSchema =
            serde_json::from_reader(&mut file).expect("failed to deserialize the schema");

        if schema != schema_file {
            eprintln!("the schema is not up to date, please run `make update-dsl-schema`");
            std::process::exit(1);
        }

        eprintln!("the DSL schema is up to date");
    }
}
