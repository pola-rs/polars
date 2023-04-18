#[cfg(feature = "highlight")]
mod highlighter;
mod interactive;
mod prompt;

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;
use polars::prelude::*;

#[global_allocator]
#[cfg(target_os = "linux")]
#[cfg(feature = "cli")]
static ALLOC: Jemalloc = Jemalloc;

use std::io::{self, BufRead};
use std::str::FromStr;

use clap::{Parser, ValueEnum};
use interactive::run_tty;
use polars::sql::SQLContext;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about)]
struct Args {
    /// Execute "COMMAND" and exit
    #[arg(short = 'c')]
    command: Option<String>,
    /// Optional query to operate on. Equivalent of `polars -c "QUERY"`
    query: Option<String>,
    #[arg(short = 'o')]
    /// Optional output mode. Defaults to 'table'
    output_mode: Option<OutputMode>,
}

#[derive(ValueEnum, Debug, Default, Clone)]
enum OutputMode {
    Csv,
    Json,
    Parquet,
    Arrow,
    #[default]
    Table,
}

impl OutputMode {
    fn execute_query(&self, query: &str, ctx: &mut SQLContext) {
        let mut execute_inner = || {
            let mut df = ctx
                .execute(query)
                .and_then(|lf| lf.collect())
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

            let w = io::stdout();
            let mut w = io::BufWriter::new(w);

            match self {
                OutputMode::Csv => CsvWriter::new(&mut w).finish(&mut df),
                OutputMode::Json => JsonWriter::new(&mut w).finish(&mut df),
                OutputMode::Parquet => ParquetWriter::new(&mut w).finish(&mut df).map(|_| ()),
                OutputMode::Arrow => IpcWriter::new(w).finish(&mut df),
                OutputMode::Table => {
                    use std::io::Write;
                    return write!(&mut w, "{df}");
                }
            }
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        };

        match execute_inner() {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }
}

impl FromStr for OutputMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "csv" => Ok(OutputMode::Csv),
            "json" => Ok(OutputMode::Json),
            "parquet" => Ok(OutputMode::Parquet),
            "arrow" => Ok(OutputMode::Arrow),
            "table" => Ok(OutputMode::Table),
            _ => Err(format!("Invalid output mode: {}", s)),
        }
    }
}

pub fn main() -> io::Result<()> {
    let args = Args::parse();
    let output_mode = args.output_mode.unwrap_or_default();

    if let Some(query) = args.command {
        let mut context = SQLContext::new();
        output_mode.execute_query(&query, &mut context);
        Ok(())
    } else if let Some(query) = args.query {
        let mut context = SQLContext::new();
        output_mode.execute_query(&query, &mut context);
        Ok(())
    } else if atty::is(atty::Stream::Stdin) {
        run_tty(output_mode)
    } else {
        run_noninteractive(output_mode)
    }
}

fn run_noninteractive(output_mode: OutputMode) -> io::Result<()> {
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

        output_mode.execute_query(query, &mut context);
    }

    Ok(())
}
