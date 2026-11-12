use std::io::{self, BufWriter};
use std::process::ExitCode;

use clap::Parser;
use polars_cli::{Args, Printer, run};

// The allocator of the Python runtime, so that both perform alike.
#[global_allocator]
static ALLOC: polars_ooc::Allocator = polars_ooc::Allocator;

fn main() -> ExitCode {
    let args = match Args::try_parse() {
        Ok(args) => args,
        // Help and version requests arrive here too, with exit code 0.
        Err(error) => {
            // The exit code stands also when the message cannot be written.
            let _ = error.print();
            return ExitCode::from(error.exit_code() as u8);
        },
    };

    // Not locked: Polars and the panic hook write to these streams from other threads.
    let stdout = BufWriter::new(io::stdout());
    let code = run(&args.command, &mut Printer::new(stdout, io::stderr()));
    ExitCode::from(code)
}
