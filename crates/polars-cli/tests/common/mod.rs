use std::env;
use std::process::Command;

/// The `polars` command, run in the crate directory with the `POLARS_*` settings of the
/// environment removed, so that they do not leak into what a test expects.
pub fn polars() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_polars"));
    command.current_dir(env!("CARGO_MANIFEST_DIR"));
    for (name, _) in env::vars_os() {
        // Names of environment variables are case-insensitive on Windows.
        if name
            .to_string_lossy()
            .to_ascii_uppercase()
            .starts_with("POLARS_")
        {
            command.env_remove(name);
        }
    }
    command
}
