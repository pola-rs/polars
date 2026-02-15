// --8<-- [start:setup]
use polars::prelude::*;
// --8<-- [end:setup]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:df]
    let df = df!(
            "foo"=> ["A", "A", "B", "B", "C"],
            "bar"=> ["k", "l", "m", "n", "o"],
            "N"=> [1, 2, 2, 4, 2],
    )?;
    println!("{}", &df);
    // --8<-- [end:df]

    // --8<-- [start:eager]
    let out = df
        .clone()
        .lazy()
        .pivot(
            Selector::ByName {
                names: [PlSmallStr::from("foo")].into(),
                strict: true,
            },
            Arc::new(df!("" => ["A", "B", "C"])?),
            Selector::ByName {
                names: [PlSmallStr::from("bar")].into(),
                strict: true,
            },
            Selector::ByName {
                names: [PlSmallStr::from("N")].into(),
                strict: true,
            },
            Expr::Agg(AggExpr::Item {
                input: Arc::new(Expr::Element),
                allow_empty: true,
            }),
            false,
            "_".into(),
        )
        .collect()?;
    println!("{}", &out);
    // --8<-- [end:eager]

    // --8<-- [start:lazy]
    let q = df.clone().lazy();
    let q2 = q.pivot(
        Selector::ByName {
            names: [PlSmallStr::from("foo")].into(),
            strict: true,
        },
        Arc::new(df!("" => ["A", "B", "C"])?),
        Selector::ByName {
            names: [PlSmallStr::from("bar")].into(),
            strict: true,
        },
        Selector::ByName {
            names: [PlSmallStr::from("N")].into(),
            strict: true,
        },
        Expr::Agg(AggExpr::Item {
            input: Arc::new(Expr::Element),
            allow_empty: true,
        }),
        false,
        "_".into(),
    );
    let out = q2.collect()?;
    println!("{}", &out);
    // --8<-- [end:lazy]

    Ok(())
}
