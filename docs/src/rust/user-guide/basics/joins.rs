use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:join]
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let df: DataFrame = df!(
        "a" => 0..8,
        "b"=> (0..8).map(|_| rng.gen::<f64>()).collect::<Vec<f64>>(),
        "d"=> [Some(1.0), Some(2.0), None, None, Some(0.0), Some(-5.0), Some(-42.), None]
    )
    .unwrap();
    let df2: DataFrame = df!(
        "x" => 0..8,
        "y"=> &["A", "A", "A", "B", "B", "C", "X", "X"],
    )
    .unwrap();
    let joined = df.join(&df2, ["a"], ["x"], JoinType::Left.into())?;
    println!("{}", joined);
    // --8<-- [end:join]

    // --8<-- [start:hstack]
    let stacked = df.hstack(df2.get_columns())?;
    println!("{}", stacked);
    // --8<-- [end:hstack]

    Ok(())
}
