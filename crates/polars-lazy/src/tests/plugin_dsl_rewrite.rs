use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

type RewriteFn = dyn Fn(&[Field]) -> PolarsResult<Expr> + Send + Sync;

struct TestRewrite {
    name: &'static str,
    f: Box<RewriteFn>,
    calls: AtomicUsize,
}

impl DslRewrite for TestRewrite {
    fn rewrite(&self, inputs: &[Field], _input_schema: &Schema) -> PolarsResult<Expr> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        (self.f)(inputs)
    }

    fn name(&self) -> PlSmallStr {
        self.name.into()
    }
}

fn test_rewrite(
    name: &'static str,
    f: impl Fn(&[Field]) -> PolarsResult<Expr> + Send + Sync + 'static,
) -> Arc<TestRewrite> {
    Arc::new(TestRewrite {
        name,
        f: Box::new(f),
        calls: AtomicUsize::new(0),
    })
}

fn call(r: &Arc<TestRewrite>, inputs: Vec<Expr>) -> Expr {
    dsl_rewrite(inputs, r.clone())
}

/// `Int64` -> `x * 2`
/// `String` ->`x.str.len_bytes()`.
fn double_or_len() -> Arc<TestRewrite> {
    test_rewrite("double_or_len", |inputs| match inputs[0].dtype() {
        DataType::Int64 => Ok(rewrite_input(0) * lit(2i64)),
        #[cfg(feature = "strings")]
        DataType::String => Ok(rewrite_input(0).str().len_bytes()),
        dt => polars_bail!(InvalidOperation: "unsupported dtype {dt}"),
    })
}

fn df() -> DataFrame {
    df!(
        "a" => [1i64, 2, 3],
        "b" => ["x", "yy", "zzz"],
        "g" => ["p", "q", "p"],
    )
    .unwrap()
}

fn assert_err_contains<T: std::fmt::Debug>(r: PolarsResult<T>, msg: &str) {
    let err = r.unwrap_err().to_string();
    assert!(err.contains(msg), "{err}");
}

/// Rewrites cannot be lowered to IR if no functionality to resolve them is given.
#[test]
fn test_rewrite_input_outside_rewrite() {
    assert!(
        load_df()
            .lazy()
            .select([Expr::RewriteInput(0) + col("a")])
            .collect_schema()
            .is_err()
    );
}

#[test]
#[cfg(feature = "strings")]
fn test_rewrite_branches_on_dtype() -> PolarsResult<()> {
    let r = double_or_len();
    let out = df()
        .lazy()
        .select([call(&r, vec![col("a")]), call(&r, vec![col("b")])])
        .collect()?;
    let expected = df!("a" => [2i64, 4, 6], "b" => [1u32, 2, 3])?;
    assert_eq!(out, expected);
    Ok(())
}

#[test]
#[cfg(feature = "dtype-extension")]
fn test_rewrite_branches_on_extension_metadata() -> PolarsResult<()> {
    use polars_core::datatypes::extension::get_extension_type_or_generic;

    let length = |unit: &str| {
        let ext = get_extension_type_or_generic("test.length", &DataType::Float64, Some(unit));
        Series::new("x".into(), [1.0f64, 2.0])
            .into_extension(ext)
            .into_column()
    };
    let df = DataFrame::new_infer_height(vec![
        length("m").with_name("m".into()),
        length("cm").with_name("cm".into()),
    ])?;

    let to_meters = test_rewrite("to_meters", |inputs| {
        let DataType::Extension(ext, _) = inputs[0].dtype() else {
            polars_bail!(InvalidOperation: "expected an extension type");
        };
        let storage = rewrite_input(0).ext().storage();
        match ext.serialize_metadata().as_deref() {
            Some("m") => Ok(storage),
            Some("cm") => Ok(storage / lit(100.0)),
            m => polars_bail!(InvalidOperation: "unknown unit {m:?}"),
        }
    });
    let out = df
        .lazy()
        .select([
            call(&to_meters, vec![col("m")]),
            call(&to_meters, vec![col("cm")]),
        ])
        .collect()?;
    let expected = df!("m" => [1.0f64, 2.0], "cm" => [0.01f64, 0.02])?;
    assert_eq!(out, expected);
    Ok(())
}

#[test]
#[cfg(feature = "cse")]
fn test_rewrite_input_used_multiple_times() -> PolarsResult<()> {
    let r = test_rewrite("twice", |_| Ok(rewrite_input(0) + rewrite_input(0)));
    let lf = df()
        .lazy()
        .select([call(&r, vec![col("a").sum()])])
        .with_comm_subexpr_elim(true);
    assert!(
        lf.explain(true)?
            .contains(polars_plan::constants::CSE_REPLACED)
    );
    assert_eq!(lf.collect()?, df!("a" => [12i64])?);
    Ok(())
}

#[test]
fn test_rewrite_in_rewrite_inputs() -> PolarsResult<()> {
    // `select` already converts its expressions more than once (e.g. for the schema), so compare
    // against a single rewrite: nesting must not add calls.
    let single = double_or_len();
    df().lazy()
        .select([call(&single, vec![col("a")])])
        .collect()?;
    let expected_calls = single.calls.load(Ordering::Relaxed);

    let inner = double_or_len();
    let outer = double_or_len();
    let out = df()
        .lazy()
        .select([call(&outer, vec![call(&inner, vec![col("a")])])])
        .collect()?;
    assert_eq!(out, df!("a" => [4i64, 8, 12])?);
    assert_eq!(inner.calls.load(Ordering::Relaxed), expected_calls);
    assert_eq!(outer.calls.load(Ordering::Relaxed), expected_calls);
    Ok(())
}

#[test]
fn test_rewrite_errors() {
    let lf = df().lazy();
    let schema_of = |e: Expr| lf.clone().select([e]).collect_schema();

    let inner = double_or_len();
    let nested = test_rewrite("nested", move |_| {
        Ok(dsl_rewrite(vec![rewrite_input(0)], inner.clone()))
    });
    assert_err_contains(
        schema_of(call(&nested, vec![col("a")])),
        "rewrite 'nested' returned an expression that contains another rewrite",
    );

    let in_eval = test_rewrite("in_eval", |_| Ok(col("l").list().eval(rewrite_input(0))));
    assert_err_contains(
        schema_of(call(&in_eval, vec![col("a")])),
        "rewrite 'in_eval' returned an expression that uses rewrite_input() inside a nested evaluation",
    );

    let multi = test_rewrite("multi", |_| Ok(all().as_expr()));
    assert_err_contains(
        schema_of(call(&multi, vec![col("a")])),
        "rewrite 'multi' must return a single expression, but it expanded into 3 expressions",
    );

    let oob = test_rewrite("oob", |_| Ok(rewrite_input(1)));
    assert_err_contains(
        schema_of(call(&oob, vec![col("a")])),
        "rewrite 'oob' returned rewrite_input(1), but it only has 1 inputs",
    );

    let failing = test_rewrite("failing", |_| polars_bail!(ComputeError: "boom"));
    assert_err_contains(
        schema_of(call(&failing, vec![col("a")])),
        "rewrite 'failing' failed",
    );
}

#[test]
fn test_rewrite_input_as_eval_subject() -> PolarsResult<()> {
    let r = test_rewrite("eval_subject", |_| {
        Ok(rewrite_input(0).list().eval(element() * lit(2i64)))
    });
    let df = df!("l" => [Series::new("".into(), [1i64, 2])])?;
    let out = df.lazy().select([call(&r, vec![col("l")])]).collect()?;
    assert_eq!(out, df!("l" => [Series::new("".into(), [2i64, 4])])?);
    Ok(())
}

#[test]
fn test_rewrite_optimizer_sees_through() -> PolarsResult<()> {
    let r = double_or_len();
    let lf = df()
        .lazy()
        .select([col("a"), col("g")])
        .filter(call(&r, vec![col("a")]).gt(lit(2i64)));
    assert!(predicate_at_scan(lf.clone()));

    let plan = lf.explain(true)?;
    assert!(plan.contains(r#"FILTER (col("a") * 2) > 2"#), "{plan}");
    assert!(!plan.contains("double_or_len"), "{plan}");
    assert_eq!(lf.collect()?, df!("a" => [2i64, 3], "g" => ["q", "p"])?);
    Ok(())
}

#[test]
fn test_rewrite_in_contexts() -> PolarsResult<()> {
    let r = double_or_len();

    let out = df()
        .lazy()
        .group_by_stable([col("g")])
        .agg([call(&r, vec![col("a")]).sum()])
        .collect()?;
    assert_eq!(out, df!("g" => ["p", "q"], "a" => [8i64, 4])?);

    let out = df()
        .lazy()
        .select([call(&r, vec![col("a")]).sum().over([col("g")])?])
        .collect()?;
    assert_eq!(out, df!("a" => [8i64, 4, 8])?);

    let df_l = df!("l" => [Series::new("".into(), [1i64, 2])])?;
    let out = df_l
        .lazy()
        .select([col("l").list().eval(call(&r, vec![element()]))])
        .collect()?;
    assert_eq!(out, df!("l" => [Series::new("".into(), [2i64, 4])])?);

    #[cfg(feature = "dtype-struct")]
    {
        let s = df()
            .lazy()
            .select([as_struct(vec![col("a")]).alias("s")])
            .select([col("s").struct_().with_fields(vec![call(
                &r,
                vec![Expr::Field(Arc::from([PlSmallStr::from("a")]))],
            )])])
            .select([col("s").struct_().field_by_name("a")])
            .collect()?;
        assert_eq!(s, df!("a" => [2i64, 4, 6])?);
    }

    Ok(())
}

#[test]
fn test_rewrite_deterministic() -> PolarsResult<()> {
    let r = double_or_len();
    let lf = df().lazy().select([call(&r, vec![col("a")])]);
    let schema = lf.clone().collect_schema()?;
    let out = lf.collect()?;
    assert_eq!(*schema, **out.schema());
    Ok(())
}
