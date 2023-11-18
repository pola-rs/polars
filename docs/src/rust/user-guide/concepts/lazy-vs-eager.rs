use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:eager]
    let df = CsvReader::from_path("docs/data/iris.csv")
        .unwrap()
        .finish()
        .unwrap();
    let mask = df.column("sepal_length")?.f64()?.gt(5.0);
    let df_small = df.filter(&mask)?;
    #[allow(deprecated)]
    let df_agg = df_small
        .group_by(["species"])?
        .select(["sepal_width"])
        .mean()?;
    println!("{}", df_agg);
    // --8<-- [end:eager]

    // --8<-- [start:lazy]
    let q = LazyCsvReader::new("docs/data/iris.csv")
        .has_header(true)
        .finish()?
        .filter(col("sepal_length").gt(lit(5)))
        .group_by(vec![col("species")])
        .agg([col("sepal_width").mean()]);
    let df = q.collect()?;
    println!("{}", df);
    // --8<-- [end:lazy]

    Ok(())
}
