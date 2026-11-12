use std::io::{self, BufWriter, IsTerminal, Read};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::error::ErrorKind;
use clap::{CommandFactory, Parser};
use polars_cli::{Args, Input, Printer, Script, exit_code, run};

// The allocator of the Python runtime, so that both perform alike.
#[global_allocator]
static ALLOC: polars_ooc::Allocator = polars_ooc::Allocator;

fn main() -> ExitCode {
    let args = match Args::try_parse() {
        Ok(args) => args,
        // Help and version requests arrive here too, with exit code 0.
        Err(error) => return report(error),
    };
    let format = args.format();
    let script = match read_script(args.input()) {
        Ok(script) => script,
        Err(error) => return report(error),
    };

    let interrupted = handle_interrupts();
    // Not locked: Polars and the panic hook write to these streams from other threads.
    let stdout = BufWriter::new(io::stdout());
    let code = run(&script, &mut Printer::new(stdout, io::stderr(), format));
    // An interrupt that arrived while the last result was printed stopped nothing.
    if code == exit_code::OK && interrupted.load(Ordering::SeqCst) {
        ExitCode::from(exit_code::INTERRUPTED)
    } else {
        ExitCode::from(code)
    }
}

fn read_script(input: Input) -> Result<Script, clap::Error> {
    match input {
        Input::Command(sql) => Ok(Script::new(sql)),
        Input::File(path) => match std::fs::read_to_string(&path) {
            Ok(text) => Ok(Script::new(text)),
            Err(error) => Err(usage_error(
                ErrorKind::Io,
                format!("cannot read '{}': {error}", path.display()),
            )),
        },
        Input::Stdin => {
            let no_sql = || usage_error(ErrorKind::MissingRequiredArgument, "no SQL was given");
            let mut stdin = io::stdin();
            if stdin.is_terminal() {
                return Err(no_sql());
            }
            let mut text = String::new();
            stdin.read_to_string(&mut text).map_err(|error| {
                usage_error(
                    ErrorKind::Io,
                    format!("cannot read standard input: {error}"),
                )
            })?;
            let script = Script::new(text);
            // Running `polars` with nothing on standard input is a mistake, not an empty script.
            if script.is_empty() {
                return Err(no_sql());
            }
            Ok(script)
        },
    }
}

/// A usage error found after the arguments were parsed, reported as clap reports its own.
fn usage_error(kind: ErrorKind, message: impl std::fmt::Display) -> clap::Error {
    Args::command().error(kind, message)
}

fn report(error: clap::Error) -> ExitCode {
    // The exit code stands also when the message cannot be written.
    let _ = error.print();
    ExitCode::from(error.exit_code() as u8)
}

/// The first interrupt asks the running statement to stop. A second one ends the process, in
/// case the statement does not react. Returns the flag that an interrupt sets.
fn handle_interrupts() -> Arc<AtomicBool> {
    use signal_hook::consts::SIGINT;

    let interrupted = Arc::new(AtomicBool::new(false));
    // The actions of a signal run in the order they were registered: the check comes first, so
    // that it sees the flag of an earlier interrupt only.
    let registered = signal_hook::flag::register_conditional_shutdown(
        SIGINT,
        i32::from(exit_code::INTERRUPTED),
        Arc::clone(&interrupted),
    )
    .and_then(|_| signal_hook::flag::register(SIGINT, Arc::clone(&interrupted)));
    // Without these two, an interrupt still stops the statement through the Polars mechanism.
    drop(registered);

    polars::error::abort::register_polars_abort_mechanism();
    interrupted
}
