#[cfg(feature = "highlight")]
mod highlighter;
mod interactive;
mod prompt;

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "linux")]
#[cfg(feature = "cli")]
static ALLOC: Jemalloc = Jemalloc;

use std::io::{self, BufRead};

use clap::Parser;
use interactive::run_tty;
use polars::sql::SQLContext;

pub(crate) fn execute_query(query: &str, ctx: &mut SQLContext) -> std::io::Result<()> {
    match ctx.execute(query).and_then(|lf| lf.collect()) {
        Ok(df) => println!("{}", df),
        Err(e) => eprintln!("Error: {}", e),
    };
    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about)]
struct Args {
    /// Execute "COMMAND" and exit
    #[arg(short = 'c')]
    command: Option<String>,
    /// Optional query to operate on. Equivalent of `polars -c "QUERY"`
    query: Option<String>,
}

pub fn main() -> io::Result<()> {
    let args = Args::parse();

    if let Some(query) = args.command {
        let mut context = SQLContext::new();
        execute_query(&query, &mut context)
    } else if let Some(query) = args.query {
        let mut context = SQLContext::new();
        execute_query(&query, &mut context)
    } else if atty::is(atty::Stream::Stdin) {
        run_tty()
    } else {
        run_noninteractive()
    }
}

fn run_noninteractive() -> io::Result<()> {
    let mut context = SQLContext::new();
    let mut input: Vec<u8> = Vec::with_capacity(1024);
    let stdin = std::io::stdin();

    loop {
        input.clear();
        stdin.lock().read_until(b';', &mut input)?;

        let query = std::str::from_utf8(&input).unwrap_or("").trim();
        if query.is_empty() {
            break;
        }

        execute_query(query, &mut context)?;
    }

    Ok(())
}
