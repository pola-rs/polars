use std::io::{self, BufWriter, Write};
use std::process::ExitCode;

use polars_cli::{HELP, Invocation, Printer, exit_code, parse_args, run};

// The allocator of the Python runtime, so that both perform alike.
#[global_allocator]
static ALLOC: polars_ooc::Allocator = polars_ooc::Allocator;

fn main() -> ExitCode {
    let invocation = match parse_args(std::env::args_os().skip(1)) {
        Ok(invocation) => invocation,
        Err(error) => {
            // The exit code stands also when stderr cannot be written to.
            let _ = writeln!(
                io::stderr(),
                "error: {error}\n\nUsage: polars -c <SQL>\nTry 'polars --help'."
            );
            return ExitCode::from(exit_code::USAGE);
        },
    };

    let stdout = BufWriter::new(io::stdout());
    let code = match invocation {
        Invocation::Help => print(stdout, HELP),
        Invocation::Version => print(stdout, concat!("polars ", env!("CARGO_PKG_VERSION"), "\n")),
        Invocation::Command(sql) => run(&sql, &mut Printer::new(stdout, io::stderr())),
    };
    ExitCode::from(code)
}

fn print(mut out: impl Write, text: &str) -> u8 {
    match out.write_all(text.as_bytes()).and_then(|()| out.flush()) {
        Ok(()) => exit_code::OK,
        Err(_) => exit_code::FAILURE,
    }
}
