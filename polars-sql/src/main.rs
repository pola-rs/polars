use std::process::exit;

use clap::Parser;
use polars::prelude::*;
use polarssql::SQLContext;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(value_parser)]
    sql: String,
}

fn run() -> PolarsResult<DataFrame> {
    let cli = Cli::parse();

    let mut context = SQLContext::new();

    println!("{:?}", cli.sql);

    let q = context.execute(&cli.sql)?;
    let out = q.limit(100).collect()?;

    Ok(out)
}

fn main() -> PolarsResult<()> {
    let out = run()?;
    println!("{}", out);

    Ok(())
}
