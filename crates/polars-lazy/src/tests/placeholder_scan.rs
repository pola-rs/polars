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
    let template = LazyFrame::placeholder("input", schema).filter(col("a").gt(lit(0)));

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
    let template = LazyFrame::placeholder("input", schema).select([col("a")]); // only select column "a"

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
    let template =
        LazyFrame::placeholder("input", schema).sort(["a"], SortMultipleOptions::default());

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
    let template =
        LazyFrame::placeholder("input", schema).with_column((col("a") * lit(2)).alias("a_doubled"));

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
    let template = LazyFrame::placeholder("input", schema).filter(col("a").gt(lit(0)));

    let df = df!["a" => [1i64, -2, 3], "b" => ["x", "y", "z"]]?;
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("input"), df.lazy());

    let bound = template.bind(bindings)?;

    // Verify the optimized plan works correctly
    let result = bound.collect()?;
    assert_eq!(result.height(), 2);

    Ok(())
}

// ==================== Phase 2: OptimizedTemplate tests ====================

#[test]
fn test_optimize_template_basic() -> PolarsResult<()> {
    let schema = make_schema();
    let lf = LazyFrame::placeholder("input", schema)
        .filter(col("a").gt(lit(0)))
        .select([col("a"), col("b")]);

    let template = lf.optimize_template()?;

    let df = df!["a" => [1i64, -2, 3], "b" => ["x", "y", "z"]]?;
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("input"), df.lazy());

    let result = template.bind_and_collect(bindings)?;
    assert_eq!(result.height(), 2);
    assert_eq!(result.width(), 2);
    let a = result.column("a")?.i64()?;
    assert_eq!(a.get(0), Some(1));
    assert_eq!(a.get(1), Some(3));

    Ok(())
}

#[test]
fn test_optimize_template_reuse() -> PolarsResult<()> {
    let schema = make_schema();
    let lf = LazyFrame::placeholder("input", schema).filter(col("a").gt(lit(0)));
    let template = lf.optimize_template()?;

    // First bind
    let df1 = df!["a" => [1i64, -2, 3], "b" => ["x", "y", "z"]]?;
    let mut b1 = HashMap::new();
    b1.insert(PlSmallStr::from("input"), df1.lazy());
    let r1 = template.bind_and_collect(b1)?;
    assert_eq!(r1.height(), 2);

    // Second bind with different data
    let df2 = df!["a" => [-1i64, 5, 10, -3], "b" => ["a", "b", "c", "d"]]?;
    let mut b2 = HashMap::new();
    b2.insert(PlSmallStr::from("input"), df2.lazy());
    let r2 = template.bind_and_collect(b2)?;
    assert_eq!(r2.height(), 2);
    let a = r2.column("a")?.i64()?;
    assert_eq!(a.get(0), Some(5));
    assert_eq!(a.get(1), Some(10));

    // Third bind
    let df3 = df!["a" => [100i64], "b" => ["only"]]?;
    let mut b3 = HashMap::new();
    b3.insert(PlSmallStr::from("input"), df3.lazy());
    let r3 = template.bind_and_collect(b3)?;
    assert_eq!(r3.height(), 1);

    Ok(())
}

#[test]
fn test_optimize_template_projection_pushdown() -> PolarsResult<()> {
    let schema = make_schema();
    // Only select "a" — projection pushdown should set output_schema on PlaceholderScan
    let lf = LazyFrame::placeholder("input", schema).select([col("a")]);
    let template = lf.optimize_template()?;

    let df = df!["a" => [1i64, 2, 3], "b" => ["x", "y", "z"]]?;
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("input"), df.lazy());

    let result = template.bind_and_collect(bindings)?;
    assert_eq!(result.width(), 1);
    assert_eq!(result.get_column_names(), &["a"]);
    assert_eq!(result.height(), 3);

    Ok(())
}

#[test]
fn test_optimize_template_with_filter() -> PolarsResult<()> {
    // Predicate pushdown places Filter above PlaceholderScan
    let schema = make_schema();
    let lf = LazyFrame::placeholder("input", schema).filter(col("a").gt(lit(2)));
    let template = lf.optimize_template()?;

    let df = df!["a" => [1i64, 2, 3, 4], "b" => ["w", "x", "y", "z"]]?;
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("input"), df.lazy());

    let result = template.bind_and_collect(bindings)?;
    assert_eq!(result.height(), 2);
    let a = result.column("a")?.i64()?;
    assert_eq!(a.get(0), Some(3));
    assert_eq!(a.get(1), Some(4));

    Ok(())
}

#[test]
fn test_optimize_template_multi_placeholder_join() -> PolarsResult<()> {
    let mut left_schema = Schema::default();
    left_schema.with_column("id".into(), DataType::Int64);
    left_schema.with_column("value".into(), DataType::Float64);

    let mut right_schema = Schema::default();
    right_schema.with_column("id".into(), DataType::Int64);
    right_schema.with_column("name".into(), DataType::String);

    let left_ph = LazyFrame::placeholder("left", left_schema);
    let right_ph = LazyFrame::placeholder("right", right_schema);

    let lf = left_ph.join(
        right_ph,
        [col("id")],
        [col("id")],
        JoinArgs::new(JoinType::Inner),
    );
    let template = lf.optimize_template()?;

    let left_df = df!["id" => [1i64, 2, 3], "value" => [10.0, 20.0, 30.0]]?;
    let right_df = df!["id" => [2i64, 3, 4], "name" => ["bob", "charlie", "dave"]]?;

    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("left"), left_df.lazy());
    bindings.insert(PlSmallStr::from("right"), right_df.lazy());

    let result = template.bind_and_collect(bindings)?;
    assert_eq!(result.height(), 2);
    assert_eq!(result.width(), 3);

    Ok(())
}

#[test]
fn test_optimize_template_schema_mismatch_error() {
    let schema = make_schema(); // a: Int64, b: String
    let lf = LazyFrame::placeholder("input", schema).select([col("a")]);
    let template = lf.optimize_template().unwrap();

    // Binding with wrong type
    let df = df!["a" => ["not_int"], "b" => ["x"]].unwrap();
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("input"), df.lazy());

    let result = template.bind_and_collect(bindings);
    assert!(result.is_err());
}

#[test]
fn test_optimize_template_missing_binding_error() {
    let schema = make_schema();
    let lf = LazyFrame::placeholder("input", schema);
    let template = lf.optimize_template().unwrap();

    let df = df!["a" => [1i64], "b" => ["x"]].unwrap();
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("wrong_name"), df.lazy());

    let result = template.bind_and_collect(bindings);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("input"));
}

#[test]
fn test_optimize_template_bind_returns_lazyframe() -> PolarsResult<()> {
    let schema = make_schema();
    let lf = LazyFrame::placeholder("input", schema).filter(col("a").gt(lit(0)));
    let template = lf.optimize_template()?;

    let df = df!["a" => [1i64, -2, 3], "b" => ["x", "y", "z"]]?;
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("input"), df.lazy());

    // Use bind() instead of bind_and_collect()
    let result = template.bind(bindings)?.collect()?;
    assert_eq!(result.height(), 2);

    Ok(())
}

#[test]
fn test_optimize_template_with_groupby() -> PolarsResult<()> {
    let schema = make_schema();
    let lf = LazyFrame::placeholder("input", schema)
        .group_by([col("b")])
        .agg([col("a").sum()]);
    let template = lf.optimize_template()?;

    let df = df!["a" => [1i64, 2, 3, 4], "b" => ["x", "y", "x", "y"]]?;
    let mut bindings = HashMap::new();
    bindings.insert(PlSmallStr::from("input"), df.lazy());

    let result = template.bind_and_collect(bindings)?;
    assert_eq!(result.height(), 2);

    Ok(())
}
