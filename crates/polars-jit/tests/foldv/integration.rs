use polars::prelude::*;
use polars_jit::{foldv, foldv_over, foldv_value};

/// A fold whose result depends on the order of every earlier row.
fn order_sensitive_fold() -> PolarsResult<Expr> {
    foldv(
        lit(0_i64).cast(DataType::Int64),
        when(Expr::Element.gt_eq(lit(1000_i64)))
            .then(Expr::Element - lit(1000_i64) + col("l"))
            .otherwise(Expr::Element * lit(2_i64) + col("l")),
    )
}

/// The same fold as a Rust loop.
fn order_sensitive_reference(values: &[i64]) -> Vec<Option<i64>> {
    let mut state = 0;
    values
        .iter()
        .map(|value| {
            state = if state >= 1000 {
                state - 1000 + value
            } else {
                state * 2 + value
            };
            Some(state)
        })
        .collect()
}

/// `count` rows as several chunks, so the fold must carry its state from one chunk to the next.
fn chunked_frame(count: usize) -> PolarsResult<(DataFrame, Vec<i64>)> {
    let values = (0..count as i64).map(|i| i % 7).collect::<Vec<_>>();
    let chunks = values
        .chunks(count / 4)
        .map(|chunk| df!("l" => chunk).map(IntoLazy::lazy))
        .collect::<PolarsResult<Vec<_>>>()?;
    let frame = concat(
        chunks,
        UnionArgs {
            rechunk: false,
            ..Default::default()
        },
    )?
    .collect_with_engine(Engine::InMemory)?
    .unwrap_single();
    assert!(frame.first_col_n_chunks() > 1);
    Ok((frame, values))
}

fn grouped_frame() -> PolarsResult<DataFrame> {
    df!(
        "g" => ["a", "b", "a", "b", "a"],
        "l" => [1_i64, 2, 3, 4, 5],
    )
}

#[test]
fn foldv_is_named_foldv_in_query_plans() -> PolarsResult<()> {
    // The plan shows the operation by name rather than as an anonymous function.
    let plan = df!("l" => [1_i64])?
        .lazy()
        .select([foldv(
            lit(0_i64).cast(DataType::Int64),
            Expr::Element + col("l"),
        )?])
        .explain(true)?;

    assert!(plan.contains(".foldv("), "{plan}");
    Ok(())
}

#[test]
fn foldv_folds_each_group_separately_in_group_by() -> PolarsResult<()> {
    // Every group starts again from the initial state and keeps its own row order.
    for engine in [Engine::InMemory, Engine::Streaming] {
        let result = grouped_frame()?
            .lazy()
            .group_by_stable([col("g")])
            .agg([
                foldv(lit(0_i64).cast(DataType::Int64), Expr::Element + col("l"))?.alias("f"),
                foldv_over(
                    col("l"),
                    lit(0_i64).cast(DataType::Int64),
                    Expr::Element + foldv_value(),
                )?
                .alias("m"),
            ])
            .collect_with_engine(engine)?
            .unwrap_single();

        for name in ["f", "m"] {
            let groups = result.column(name)?.list()?;
            assert_eq!(
                groups.get_as_series(0).unwrap().i64()?.to_vec(),
                [Some(1), Some(4), Some(9)]
            );
            assert_eq!(
                groups.get_as_series(1).unwrap().i64()?.to_vec(),
                [Some(2), Some(6)]
            );
        }
    }
    Ok(())
}

#[test]
fn foldv_takes_each_groups_own_initial_state() -> PolarsResult<()> {
    // An initial state computed per group starts that group's fold.
    let result = grouped_frame()?
        .lazy()
        .group_by_stable([col("g")])
        .agg([foldv(col("l").first() * lit(100_i64), Expr::Element + col("l"))?.alias("f")])
        .collect_with_engine(Engine::InMemory)?
        .unwrap_single();

    let groups = result.column("f")?.list()?;
    assert_eq!(
        groups.get_as_series(0).unwrap().i64()?.to_vec(),
        [Some(101), Some(104), Some(109)]
    );
    assert_eq!(
        groups.get_as_series(1).unwrap().i64()?.to_vec(),
        [Some(202), Some(206)]
    );
    Ok(())
}

#[test]
fn foldv_folds_each_window_separately_with_over() -> PolarsResult<()> {
    // Each row receives the state of its own group's fold, in the frame's row order.
    for engine in [Engine::InMemory, Engine::Streaming] {
        let result = grouped_frame()?
            .lazy()
            .with_columns([
                foldv(lit(0_i64).cast(DataType::Int64), Expr::Element + col("l"))?
                    .over([col("g")])?
                    .alias("r"),
            ])
            .collect_with_engine(engine)?
            .unwrap_single();

        assert_eq!(
            result.column("r")?.i64()?.to_vec(),
            [Some(1), Some(2), Some(4), Some(6), Some(9)]
        );
    }
    Ok(())
}

#[test]
fn foldv_adds_a_column_with_with_columns() -> PolarsResult<()> {
    // The fold has one value per row, so it can sit beside the frame's other columns.
    let result = grouped_frame()?
        .lazy()
        .with_columns([
            foldv(lit(0_i64).cast(DataType::Int64), Expr::Element + col("l"))?.alias("r"),
        ])
        .collect_with_engine(Engine::InMemory)?
        .unwrap_single();

    assert_eq!(result.get_column_names(), ["g", "l", "r"]);
    assert_eq!(
        result.column("r")?.i64()?.to_vec(),
        [Some(1), Some(3), Some(6), Some(10), Some(15)]
    );
    Ok(())
}

#[test]
fn foldv_folds_only_the_rows_remaining_after_a_filter() -> PolarsResult<()> {
    // A filter before the fold removes rows from the fold, not just from its output.
    for engine in [Engine::InMemory, Engine::Streaming] {
        let result = grouped_frame()?
            .lazy()
            .filter(col("g").eq(lit("a")))
            .select([foldv(lit(0_i64).cast(DataType::Int64), Expr::Element + col("l"))?.alias("r")])
            .collect_with_engine(engine)?
            .unwrap_single();

        assert_eq!(
            result.column("r")?.i64()?.to_vec(),
            [Some(1), Some(4), Some(9)]
        );
    }
    Ok(())
}

#[test]
fn foldv_carries_the_state_across_chunks_and_streaming_morsels() -> PolarsResult<()> {
    // Large chunked input is split into several morsels by the streaming engine; the fold must
    // still see every row once, in order, with one carried state.
    let (frame, values) = chunked_frame(200_000)?;
    for engine in [Engine::InMemory, Engine::Streaming] {
        let result = frame
            .clone()
            .lazy()
            .select([order_sensitive_fold()?.alias("r")])
            .collect_with_engine(engine)?
            .unwrap_single();

        assert_eq!(
            result.column("r")?.i64()?.to_vec(),
            order_sensitive_reference(&values)
        );
    }
    Ok(())
}
