use polars::prelude::*;

#[cfg(feature = "binary")]
mod binary {
    use clap::Parser;
    use polarssql::SQLContext;

    use super::*;

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
}

fn main() -> PolarsResult<()> {
    #[cfg(feature = "binary")]
    {
        let out = run()?;
        println!("{}", out);
    }
    Ok(())
}
