fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:mansum]
    use polars::lazy::dsl::sum_horizontal;
    use polars::prelude::*;

    let df = df!(
        "label" => ["foo", "bar", "spam"],
        "a" => [1, 2, 3],
        "b" => [10, 20, 30],
    )?;

    let result = df
        .clone()
        .lazy()
        .select([
            fold_exprs(
                lit(0),
                PlanCallback::new(|(acc, val)| &acc + &val),
                [col("a"), col("b")],
                false,
                None,
            )
            .alias("sum_fold"),
            sum_horizontal([col("a"), col("b")], true)?.alias("sum_horz"),
        ])
        .collect()?;

    println!("{result:?}");
    // --8<-- [end:mansum]

    // --8<-- [start:mansum-explicit]
    let acc = lit(0);
    let f = |acc: Expr, val: Expr| acc + val;

    let result = df
        .clone()
        .lazy()
        .select([
            f(f(acc, col("a")), col("b")),
            fold_exprs(
                lit(0),
                PlanCallback::new(|(acc, val)| &acc + &val),
                [col("a"), col("b")],
                false,
                None,
            )
            .alias("sum_fold"),
        ])
        .collect()?;

    println!("{result:?}");
    // --8<-- [end:mansum-explicit]

    // --8<-- [start:manprod]
    let result = df
        .clone()
        .lazy()
        .select([fold_exprs(
            lit(0),
            PlanCallback::new(|(acc, val)| &acc * &val),
            [col("a"), col("b")],
            false,
            None,
        )
        .alias("prod")])
        .collect()?;

    println!("{result:?}");
    // --8<-- [end:manprod]

    // --8<-- [start:manprod-fixed]
    let result = df
        .lazy()
        .select([fold_exprs(
            lit(1),
            PlanCallback::new(|(acc, val)| &acc * &val),
            [col("a"), col("b")],
            false,
            None,
        )
        .alias("prod")])
        .collect()?;

    println!("{result:?}");
    // --8<-- [end:manprod-fixed]

    // --8<-- [start:conditional]
    let df = df!(
        "a" => [1, 2, 3],
        "b" => [0, 1, 2],
    )?;

    let result = df
        .lazy()
        .filter(fold_exprs(
            lit(true),
            PlanCallback::new(|(acc, val)| &acc & &val),
            [col("*").gt(1)],
            false,
            None,
        ))
        .collect()?;

    println!("{result:?}");
    // --8<-- [end:conditional]

    // --8<-- [start:string]
    let df = df!(
        "a" => ["a", "b", "c"],
        "b" => [1, 2, 3],
    )?;

    let result = df
        .lazy()
        .select([concat_str([col("a"), col("b")], "", false)])
        .collect()?;
    println!("{result:?}");
    // --8<-- [end:string]

    Ok(())
}
