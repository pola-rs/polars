use super::*;

#[test]
fn test_lazy_window_functions() {
    let df = df! {
        "groups" => &[1, 1, 2, 2, 1, 2, 3, 3, 1],
        "values" => &[1, 2, 3, 4, 5, 6, 7, 8, 8]
    }
    .unwrap();

    // sums
    // 1 => 16
    // 2 => 13
    // 3 => 15
    let correct = [16, 16, 13, 13, 16, 13, 15, 15, 16]
        .iter()
        .copied()
        .map(Some)
        .collect::<Vec<_>>();

    // test if groups is available after projection pushdown.
    let _ = df
        .clone()
        .lazy()
        .select(&[avg("values").over([col("groups")]).alias("part")])
        .collect()
        .unwrap();
    // test if partition aggregation is correct
    let out = df
        .lazy()
        .select([col("groups"), sum("values").over([col("groups")])])
        .collect()
        .unwrap();
    assert_eq!(
        Vec::from(out.select_at_idx(1).unwrap().i32().unwrap()),
        correct
    );
}

#[test]
fn test_shift_and_fill_window_function() -> Result<()> {
    let df = fruits_cars();

    // a ternary expression with a final list aggregation
    let out1 = df
        .clone()
        .lazy()
        .select([
            col("fruits"),
            col("B")
                .shift_and_fill(-1, lit(-1))
                .list()
                .over([col("fruits")]),
        ])
        .collect()?;

    // same expression, no final list aggregation
    let out2 = df
        .lazy()
        .select([
            col("fruits"),
            col("B")
                .shift_and_fill(-1, lit(-1))
                .list()
                .over([col("fruits")]),
        ])
        .collect()?;

    assert!(out1.frame_equal(&out2));

    Ok(())
}

#[test]
fn test_exploded_window_function() -> Result<()> {
    let df = fruits_cars();

    let out = df
        .clone()
        .lazy()
        .sort("fruits", false)
        .select([
            col("fruits"),
            col("B")
                .shift(1)
                .list()
                .over([col("fruits")])
                .explode()
                .alias("shifted"),
        ])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("shifted")?.i32()?),
        &[None, Some(3), None, Some(5), Some(4)]
    );

    // this tests if cast succeeds in aggregation context
    // we implicitly also test that a literal does not upcast a column
    let out = df
        .lazy()
        .sort("fruits", false)
        .select([
            col("fruits"),
            col("B")
                .shift_and_fill(1, lit(-1.0f32))
                .list()
                .over([col("fruits")])
                .explode()
                .alias("shifted"),
        ])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("shifted")?.f32()?),
        &[Some(-1.0), Some(3.0), Some(-1.0), Some(5.0), Some(4.0)]
    );
    Ok(())
}

#[test]
fn test_reverse_in_groups() -> Result<()> {
    let df = fruits_cars();

    let out = df
        .lazy()
        .sort("fruits", false)
        .select([
            col("B"),
            col("fruits"),
            col("B").reverse().over([col("fruits")]).alias("rev"),
        ])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("rev")?.i32()?),
        &[Some(2), Some(3), Some(1), Some(4), Some(5)]
    );
    Ok(())
}

#[test]
fn test_sort_by_in_groups() -> Result<()> {
    let df = fruits_cars();

    let out = df
        .lazy()
        .sort("cars", false)
        .select([
            col("fruits"),
            col("cars"),
            col("A")
                .sort_by([col("B")], [false])
                .list()
                .over([col("cars")])
                .explode()
                .alias("sorted_A_by_B"),
        ])
        .collect()?;

    assert_eq!(
        Vec::from(out.column("sorted_A_by_B")?.i32()?),
        &[Some(2), Some(5), Some(4), Some(3), Some(1)]
    );
    Ok(())
}
#[test]
fn test_literal_window_fn() -> Result<()> {
    let df = df![
        "chars" => ["a", "a", "b"]
    ]?;

    let out = df
        .lazy()
        .select([lit(1)
            .cumsum(false)
            .list()
            .over([col("chars")])
            .alias("foo")])
        .collect()?;

    let out = out.column("foo")?;
    assert!(matches!(out.dtype(), DataType::List(_)));
    let flat = out.explode()?;
    let flat = flat.i32()?;
    assert_eq!(
        Vec::from(flat),
        &[Some(1), Some(2), Some(1), Some(2), Some(1)]
    );

    Ok(())
}

#[test]
fn test_window_mapping() -> Result<()> {
    let df = fruits_cars();

    // no aggregation
    let out = df
        .clone()
        .lazy()
        .select([col("A").over([col("fruits")])])
        .collect()?;

    assert!(out.column("A")?.series_equal(df.column("A")?));

    let out = df
        .clone()
        .lazy()
        .select([col("A"), lit(0).over([col("fruits")])])
        .collect()?;

    assert_eq!(out.shape(), (5, 2));

    let out = df
        .clone()
        .lazy()
        .select([(lit(10) + col("A")).alias("foo").over([col("fruits")])])
        .collect()?;

    let expected = Series::new("foo", [11, 12, 13, 14, 15]);
    assert!(out.column("foo")?.series_equal(&expected));

    let out = df
        .clone()
        .lazy()
        .select([
            col("fruits"),
            col("B"),
            col("A"),
            (col("B").sum() + col("A"))
                .alias("foo")
                .over([col("fruits")]),
        ])
        .collect()?;
    let expected = Series::new("foo", [11, 12, 8, 9, 15]);
    assert!(out.column("foo")?.series_equal(&expected));

    let out = df
        .clone()
        .lazy()
        .select([
            col("fruits"),
            col("A"),
            col("B"),
            (col("B").shift(1) - col("A"))
                .alias("foo")
                .over([col("fruits")]),
        ])
        .collect()?;
    let expected = Series::new("foo", [None, Some(3), None, Some(-1), Some(-1)]);
    assert!(out.column("foo")?.series_equal_missing(&expected));

    // now sorted
    // this will tragger a fast path
    let df = df.sort(["fruits"], vec![false])?;

    let out = df
        .clone()
        .lazy()
        .select([(lit(10) + col("A")).alias("foo").over([col("fruits")])])
        .collect()?;
    let expected = Series::new("foo", [13, 14, 11, 12, 15]);
    assert!(out.column("foo")?.series_equal(&expected));

    let out = df
        .clone()
        .lazy()
        .select([
            col("fruits"),
            col("B"),
            col("A"),
            (col("B").sum() + col("A"))
                .alias("foo")
                .over([col("fruits")]),
        ])
        .collect()?;

    let expected = Series::new("foo", [8, 9, 11, 12, 15]);
    assert!(out.column("foo")?.series_equal(&expected));

    let out = df
        .clone()
        .lazy()
        .select([
            col("fruits"),
            col("A"),
            col("B"),
            (col("B").shift(1) - col("A"))
                .alias("foo")
                .over([col("fruits")]),
        ])
        .collect()?;

    let expected = Series::new("foo", [None, Some(-1), None, Some(3), Some(-1)]);
    assert!(out.column("foo")?.series_equal_missing(&expected));

    Ok(())
}

#[test]
fn test_window_exprs_in_binary_exprs() -> Result<()> {
    let df = df![
        "value" => 0..8,
        "cat" => [0, 0, 0, 0, 1, 1, 1, 1]
    ]?
    .lazy()
    .with_columns([
        (col("value") - col("value").mean().over([col("cat")]))
            .cast(DataType::Int32)
            .alias("centered"),
        (col("value") - col("value").std().over([col("cat")]))
            .cast(DataType::Int32)
            .alias("scaled"),
        ((col("value") - col("value").mean().over([col("cat")]))
            / col("value").std().over([col("cat")]))
        .cast(DataType::Int32)
        .alias("stdized"),
        ((col("value") - col("value").mean()).over([col("cat")]) / col("value").std())
            .cast(DataType::Int32)
            .alias("stdized2"),
        ((col("value") - col("value").mean()) / col("value").std())
            .over([col("cat")])
            .cast(DataType::Int32)
            .alias("stdized3"),
    ])
    .sum()
    .collect()?;

    let expected = df![
        "value" => [28],
        "cat" => [4],
        "centered" => [0],
        "scaled" => [14],
        "stdized" => [0],
        "stdized2" => [0],
        "stdized3" => [0]
    ]?;

    assert!(df.frame_equal(&expected));

    Ok(())
}

#[test]
fn test_window_exprs_any_all() -> Result<()> {
    let df = df![
        "var1"=> ["A", "B", "C", "C", "D", "D", "E", "E"],
        "var2"=> [false, true, false, false, false, true, true, true],
    ]?
    .lazy()
    .select([
        col("var2").any().over([col("var1")]).alias("any"),
        col("var2").all().over([col("var1")]).alias("all"),
    ])
    .collect()?;

    let expected = df![
        "any" => [false, true, false, false, true, true, true, true],
        "all" => [false, true, false, false, false, false, true, true],
    ]?;
    assert!(df.frame_equal(&expected));
    Ok(())
}

#[test]
fn test_window_naive_any() -> Result<()> {
    let df = df![
        "row_id" => [0, 0, 1, 1, 1],
        "boolvar" => [true, false, true, false, false]
    ]?;

    let df = df
        .lazy()
        .with_column(
            col("boolvar")
                .sum()
                .gt(lit(0))
                .over([col("row_id")])
                .alias("res"),
        )
        .collect()?;

    let res = df.column("res")?;
    assert_eq!(res.sum::<usize>(), Some(5));
    Ok(())
}
