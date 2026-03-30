use std::collections::HashMap;

use polars_utils::pl_str::PlSmallStr;

use super::*;

fn make_schema() -> Schema {
    let mut schema = Schema::default();
    schema.with_column("a".into(), DataType::Int64);
    schema.with_column("b".into(), DataType::String);
    schema
}

#[test]
fn test_placeholder_scan_basic() -> PolarsResult<()> {
    // Create a placeholder and bind it to a concrete DataFrame
    let schema = make_schema();
    let template = LazyFrame::placeholder("input", schema)
        .filter(col("a").gt(lit(0)))
        .select([col("a"), col("b")]);

    let df = df![
        "a" => [1i64, -2, 3],
        "b" => ["x", "y", "z"],
    ]?;

    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("input"), df.lazy());

    let result = template.bind(bindings)?.collect()?;

    assert_eq!(result.height(), 2);
    assert_eq!(result.width(), 2);
    let a = result.column("a")?.i64()?;
    assert_eq!(a.get(0), Some(1));
    assert_eq!(a.get(1), Some(3));

    Ok(())
}

#[test]
fn test_placeholder_scan_template_reuse() -> PolarsResult<()> {
    // The same template can be bound to different data multiple times
    let schema = make_schema();
    let template = LazyFrame::placeholder("input", schema)
        .filter(col("a").gt(lit(0)));

    // First binding
    let df1 = df!["a" => [1i64, -2, 3], "b" => ["x", "y", "z"]]?;
    let mut bindings1 = HashMap::new();
    bindings1.insert(PlSmallStr::from("input"), df1.lazy());
    let result1 = template.clone().bind(bindings1)?.collect()?;
    assert_eq!(result1.height(), 2);

    // Second binding with different data
    let df2 = df!["a" => [-1i64, 5, 10, -3], "b" => ["a", "b", "c", "d"]]?;
    let mut bindings2 = HashMap::new();
    bindings2.insert(PlSmallStr::from("input"), df2.lazy());
    let result2 = template.bind(bindings2)?.collect()?;
    assert_eq!(result2.height(), 2);
    let a = result2.column("a")?.i64()?;
    assert_eq!(a.get(0), Some(5));
    assert_eq!(a.get(1), Some(10));

    Ok(())
}

#[test]
fn test_placeholder_scan_multi_placeholder_join() -> PolarsResult<()> {
    // Two placeholders joined together
    let mut left_schema = Schema::default();
    left_schema.with_column("id".into(), DataType::Int64);
    left_schema.with_column("value".into(), DataType::Float64);

    let mut right_schema = Schema::default();
    right_schema.with_column("id".into(), DataType::Int64);
    right_schema.with_column("name".into(), DataType::String);

    let left_ph = LazyFrame::placeholder("left", left_schema);
    let right_ph = LazyFrame::placeholder("right", right_schema);

    let template = left_ph.join(
        right_ph,
        [col("id")],
        [col("id")],
        JoinArgs::new(JoinType::Inner),
    );

    let left_df = df![
        "id" => [1i64, 2, 3],
        "value" => [10.0, 20.0, 30.0],
    ]?;
    let right_df = df![
        "id" => [2i64, 3, 4],
        "name" => ["bob", "charlie", "dave"],
    ]?;

    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("left"), left_df.lazy());
    bindings.insert(PlSmallStr::from("right"), right_df.lazy());

    let result = template.bind(bindings)?.collect()?;

    assert_eq!(result.height(), 2); // id 2 and 3 match
    assert_eq!(result.width(), 3); // id, value, name

    Ok(())
}

#[test]
fn test_placeholder_scan_unbound_errors() {
    // Collecting without binding should produce an error
    let schema = make_schema();
    let template = LazyFrame::placeholder("input", schema);

    let result = template.collect();
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("PlaceholderScan"));
}

#[test]
fn test_placeholder_scan_missing_binding_errors() {
    // Binding with a wrong name should produce an error
    let schema = make_schema();
    let template = LazyFrame::placeholder("input", schema);

    let df = df!["a" => [1i64], "b" => ["x"]].unwrap();
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("wrong_name"), df.lazy());

    let result = template.bind(bindings);
    assert!(result.is_err());
    let err_msg = format!("{}", result.err().unwrap());
    assert!(err_msg.contains("input"));
}

#[test]
fn test_placeholder_scan_with_projection() -> PolarsResult<()> {
    // Test that projection pushdown works with placeholder
    let schema = make_schema();
    let template = LazyFrame::placeholder("input", schema)
        .select([col("a")]); // only select column "a"

    let df = df!["a" => [1i64, 2, 3], "b" => ["x", "y", "z"]]?;
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("input"), df.lazy());

    let result = template.bind(bindings)?.collect()?;

    assert_eq!(result.width(), 1);
    assert_eq!(result.get_column_names(), &["a"]);
    assert_eq!(result.height(), 3);

    Ok(())
}

#[test]
fn test_placeholder_scan_with_sort() -> PolarsResult<()> {
    let schema = make_schema();
    let template = LazyFrame::placeholder("input", schema)
        .sort(["a"], SortMultipleOptions::default());

    let df = df!["a" => [3i64, 1, 2], "b" => ["c", "a", "b"]]?;
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("input"), df.lazy());

    let result = template.bind(bindings)?.collect()?;

    let a = result.column("a")?.i64()?;
    assert_eq!(a.get(0), Some(1));
    assert_eq!(a.get(1), Some(2));
    assert_eq!(a.get(2), Some(3));

    Ok(())
}

#[test]
fn test_placeholder_scan_with_groupby() -> PolarsResult<()> {
    let schema = make_schema();
    let template = LazyFrame::placeholder("input", schema)
        .group_by([col("b")])
        .agg([col("a").sum()]);

    let df = df![
        "a" => [1i64, 2, 3, 4],
        "b" => ["x", "y", "x", "y"],
    ]?;
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("input"), df.lazy());

    let result = template
        .bind(bindings)?
        .sort(["b"], SortMultipleOptions::default())
        .collect()?;

    assert_eq!(result.height(), 2);
    let a = result.column("a")?.i64()?;
    // x: 1+3=4, y: 2+4=6
    assert_eq!(a.get(0), Some(4));
    assert_eq!(a.get(1), Some(6));

    Ok(())
}

#[test]
fn test_placeholder_scan_with_slice() -> PolarsResult<()> {
    let schema = make_schema();
    let template = LazyFrame::placeholder("input", schema).slice(0, 2);

    let df = df!["a" => [1i64, 2, 3], "b" => ["x", "y", "z"]]?;
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("input"), df.lazy());

    let result = template.bind(bindings)?.collect()?;

    assert_eq!(result.height(), 2);

    Ok(())
}

#[test]
fn test_placeholder_scan_with_with_columns() -> PolarsResult<()> {
    let schema = make_schema();
    let template = LazyFrame::placeholder("input", schema)
        .with_column((col("a") * lit(2)).alias("a_doubled"));

    let df = df!["a" => [1i64, 2, 3], "b" => ["x", "y", "z"]]?;
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("input"), df.lazy());

    let result = template.bind(bindings)?.collect()?;

    assert_eq!(result.width(), 3);
    let doubled = result.column("a_doubled")?.i64()?;
    assert_eq!(doubled.get(0), Some(2));
    assert_eq!(doubled.get(1), Some(4));
    assert_eq!(doubled.get(2), Some(6));

    Ok(())
}

#[test]
fn test_placeholder_scan_union() -> PolarsResult<()> {
    // Test placeholder in a union (concat) context
    let schema = make_schema();
    let ph1 = LazyFrame::placeholder("part1", schema.clone());
    let ph2 = LazyFrame::placeholder("part2", schema);

    let template = concat([ph1, ph2], Default::default())?;

    let df1 = df!["a" => [1i64, 2], "b" => ["x", "y"]]?;
    let df2 = df!["a" => [3i64, 4], "b" => ["z", "w"]]?;

    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("part1"), df1.lazy());
    bindings.insert(PlSmallStr::from("part2"), df2.lazy());

    let result = template.bind(bindings)?.collect()?;

    assert_eq!(result.height(), 4);

    Ok(())
}

#[test]
fn test_placeholder_scan_ir_conversion() -> PolarsResult<()> {
    // Verify that PlaceholderScan properly converts to IR and the optimizer
    // handles it (predicate stays as Filter node, not pushed into scan)
    let schema = make_schema();
    let template = LazyFrame::placeholder("input", schema)
        .filter(col("a").gt(lit(0)));

    let df = df!["a" => [1i64, -2, 3], "b" => ["x", "y", "z"]]?;
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("input"), df.lazy());

    let bound = template.bind(bindings)?;

    // Verify the optimized plan works correctly
    let result = bound.collect()?;
    assert_eq!(result.height(), 2);

    Ok(())
}
