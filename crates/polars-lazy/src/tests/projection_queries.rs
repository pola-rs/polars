use polars_ops::frame::JoinCoalesce;

use super::*;

#[test]
fn test_join_suffix_and_drop() -> PolarsResult<()> {
    let weight = df![
        "id" => [1, 2, 3, 4, 5, 0],
        "wgt" => [4.32, 5.23, 2.33, 23.399, 392.2, 0.0]
    ]?
    .lazy();

    let ped = df![
        "id"=> [1, 2, 3, 4, 5],
        "sireid"=> [0, 0, 1, 3, 3]
    ]?
    .lazy();

    let sumry = weight
        .clone()
        .filter(col("id").eq(lit(2i32)))
        .inner_join(ped, "id", "id");

    let out = sumry
        .join_builder()
        .with(weight)
        .left_on([col("sireid")])
        .right_on([col("id")])
        .suffix("_sire")
        .finish()
        .drop(cols(["sireid"]))
        .collect()?;

    assert_eq!(out.shape(), (1, 3));

    Ok(())
}

#[test]
#[cfg(feature = "cross_join")]
fn test_cross_join_pd() -> PolarsResult<()> {
    let food = df![
        "name"=> ["Omelette", "Fried Egg"],
        "price" => [8, 5]
    ]?;

    let drink = df![
        "name" => ["Orange Juice", "Tea"],
        "price" => [5, 4]
    ]?;

    let q = food.lazy().cross_join(drink.lazy(), None).select([
        col("name").alias("food"),
        col("name_right").alias("beverage"),
        (col("price") + col("price_right")).alias("total"),
    ]);

    let out = q.collect()?;
    let expected = df![
        "food" => ["Omelette", "Omelette", "Fried Egg", "Fried Egg"],
        "beverage" => ["Orange Juice", "Tea", "Orange Juice", "Tea"],
        "total" => [13, 12, 10, 9]
    ]?;

    assert!(out.equals(&expected));
    Ok(())
}

#[test]
fn test_row_number_pd() -> PolarsResult<()> {
    let df = df![
        "x" => [1, 2, 3],
        "y" => [3, 2, 1],
    ]?;

    let df = df
        .lazy()
        .with_row_index("index", None)
        .select([col("index"), col("x") * lit(3i32)])
        .collect()?;

    let expected = df![
        "index" => [0 as IdxSize, 1, 2],
        "x" => [3i32, 6, 9]
    ]?;

    assert!(df.equals(&expected));

    Ok(())
}

#[test]
#[cfg(feature = "cse")]
fn scan_join_same_file() -> PolarsResult<()> {
    let lf = LazyCsvReader::new(PlPath::new(FOODS_CSV)).finish()?;

    for cse in [true, false] {
        let partial = lf.clone().select([col("category")]).limit(5);
        let q = lf
            .clone()
            .join(
                partial,
                [col("category")],
                [col("category")],
                JoinType::Inner.into(),
            )
            .with_comm_subplan_elim(cse);
        let out = q.collect()?;
        assert_eq!(
            out.get_column_names(),
            &["category", "calories", "fats_g", "sugars_g"]
        );
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "regex", feature = "concat_str"))]
fn concat_str_regex_expansion() -> PolarsResult<()> {
    let df = df![
        "a"=> [1, 1, 1],
        "b_a_1"=> ["a--", "", ""],
        "b_a_2"=> ["", "b--", ""],
        "b_a_3"=> ["", "", "c--"]
    ]?
    .lazy();
    let out = df
        .select([concat_str([col(r"^b_a_\d$")], ";", false).alias("concatenated")])
        .collect()?;
    let s = out.column("concatenated")?;
    assert_eq!(
        s,
        &Column::new("concatenated".into(), ["a--;;", ";b--;", ";;c--"])
    );

    Ok(())
}

#[test]
fn test_coalesce_toggle_projection_pushdown() -> PolarsResult<()> {
    // Test that the optimizer toggle coalesce to true if the non-coalesced column isn't used.
    let q1 = df!["a" => [1],
        "b" => [2]
    ]?
    .lazy();

    let q2 = df!["a" => [1],
        "c" => [2]
    ]?
    .lazy();

    let plan = q1
        .join(
            q2,
            [col("a")],
            [col("a")],
            JoinArgs {
                how: JoinType::Left,
                coalesce: JoinCoalesce::KeepColumns,
                ..Default::default()
            },
        )
        .select([col("a"), col("b")])
        .to_alp_optimized()?;

    let node = plan.lp_top;
    let lp_arena = plan.lp_arena;

    assert!(lp_arena.iter(node).all(|(_, plan)| match plan {
        IR::Join { options, .. } => options.args.should_coalesce(),
        _ => true,
    }));

    Ok(())
}

#[test]
fn test_select_hconcat_pushdown_non_strict_25263() -> PolarsResult<()> {
    let df_a = df![
        "a" => [1, 2, 2],
        "b" => [4, 5, 6],
    ]?
    .lazy();

    let df_b = df![
        "d" => [1, 2],
    ]?
    .lazy();

    // not strict: we read a single column from `df_a` to ensure that the concat output
    // has the correct height
    let lf = concat_lf_horizontal([df_a, df_b], UnionArgs::default())?.select([col("d")]);
    let plan = lf.clone().to_alp_optimized()?;

    let node = plan.lp_top;
    let lp_arena = plan.lp_arena;

    assert!(lp_arena.iter(node).all(|(_, plan)| match plan {
        IR::DataFrameScan {
            schema,
            output_schema,
            ..
        } => {
            // make sure that for `df_a` we apply a projection pushdown to only read a single column
            if schema.contains("a") {
                assert_eq!(output_schema.as_ref().unwrap().len(), 1);
            }
            true
        },
        _ => true,
    }));

    let out = lf.collect()?;
    assert_eq!(
        out,
        df![
            "d" => [Some(1), Some(2), None]
        ]?
    );

    Ok(())
}

#[test]
fn test_select_hconcat_pushdown_strict_25263() -> PolarsResult<()> {
    let df_a = df![
        "a" => [1, 2],
        "b" => [4, 5],
    ]?
    .lazy();

    let df_b = df![
        "d" => [1, 2],
    ]?
    .lazy();

    // strict: we don't read any columns from `df_a`
    let lf = concat_lf_horizontal(
        [df_a, df_b],
        UnionArgs {
            strict: true,
            ..Default::default()
        },
    )?
    .select([col("d")]);
    let plan = lf.clone().to_alp_optimized()?;

    let node = plan.lp_top;
    let lp_arena = plan.lp_arena;

    assert!(lp_arena.iter(node).all(|(_, plan)| match plan {
        IR::DataFrameScan { schema, .. } => {
            // make sure that we don't read any columns from `df_a`
            if schema.contains("a") {
                panic!("should not have read any columns from `df_a`");
            }
            true
        },
        _ => true,
    }));

    let out = lf.collect()?;
    assert_eq!(
        out,
        df![
            "d" => [Some(1), Some(2)]
        ]?
    );

    Ok(())
}
