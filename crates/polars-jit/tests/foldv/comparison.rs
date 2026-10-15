//! Comparison tests: foldv must behave exactly like native Polars.
//!
//! Each test runs a fold body through the JIT and through native Polars, which evaluates the same
//! body one row at a time while carrying the state between rows. The two must agree on dtype,
//! values, nulls, NaN, signed zero, and errors, and the planned schema must match the result.
//! Add comparison tests for every operation the JIT supports; a difference from Polars is a bug
//! unless it is documented as intentional.

use polars::chunked_array::cast::CastOptions;
use polars::prelude::*;
use polars_jit::foldv;
use polars_plan::plans::DynLiteralValue;

const STATE: &str = "__foldv_reference_state__";
const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;

type Body<'a> = &'a dyn Fn(Expr) -> Expr;
/// A named binary operation applied to two expressions.
type OperatorCase = (&'static str, fn(Expr, Expr) -> Expr);
/// A named fold body.
type BodyCase = (&'static str, fn(Expr) -> Expr);
/// A named fold body, its input, and the dtype Polars gives an untyped null state.
type InferenceCase = (&'static str, DataFrame, fn(Expr) -> Expr, DataType);
/// A named fold body and the state dtype it produces.
type TypedBodyCase = (&'static str, fn(Expr) -> Expr, DataType);
/// A named fold body that captures values.
type CapturingBodyCase = (&'static str, Box<dyn Fn(Expr) -> Expr>);

/// Evaluate `body` with native Polars one row at a time, carrying the state between rows.
fn reference(df: &DataFrame, body: Body<'_>, initial_state: Expr) -> PolarsResult<Series> {
    let mut state = DataFrame::empty()
        .lazy()
        .select([initial_state.alias(STATE)])
        .collect()?
        .column(STATE)?
        .as_materialized_series()
        .clone();
    let dtype = state.dtype().clone();
    let mut output = Series::new_empty("reference".into(), &dtype);
    for row in 0..df.height() {
        let mut frame = df.slice(row as i64, 1);
        frame.with_column(state.into_column())?;
        state = frame
            .lazy()
            .select([body(col(STATE)).strict_cast(dtype.clone()).alias(STATE)])
            .collect()?
            .column(STATE)?
            .as_materialized_series()
            .clone();
        output.append(&state)?;
    }
    Ok(output)
}

/// Equal values, where NaN equals NaN and signed zeros differ.
fn same(left: &AnyValue<'_>, right: &AnyValue<'_>) -> bool {
    match (left, right) {
        (AnyValue::Float32(l), AnyValue::Float32(r)) => {
            (l.is_nan() && r.is_nan()) || l.to_bits() == r.to_bits()
        },
        (AnyValue::Float64(l), AnyValue::Float64(r)) => {
            (l.is_nan() && r.is_nan()) || l.to_bits() == r.to_bits()
        },
        (AnyValue::Null, AnyValue::Null) => true,
        _ => left == right,
    }
}

/// The underlying error, without the context Polars adds while it propagates.
fn root_error(error: &PolarsError) -> &PolarsError {
    match error {
        PolarsError::Context { error, .. } | PolarsError::ExprContext { error, .. } => {
            root_error(error)
        },
        error => error,
    }
}

fn assert_matches_polars(case: &str, df: &DataFrame, body: Body<'_>, initial_state: Expr) {
    assert_matches_polars_with(case, df, body, initial_state.clone(), initial_state);
}

/// Compare foldv started from `initial_state` with Polars started from `reference_state`.
fn assert_matches_polars_with(
    case: &str,
    df: &DataFrame,
    body: Body<'_>,
    initial_state: Expr,
    reference_state: Expr,
) {
    let expected = reference(df, body, reference_state);
    let query = foldv(initial_state, body(Expr::Element))
        .map(|expression| df.clone().lazy().select([expression.alias("result")]));
    let actual = query
        .clone()
        .and_then(|query| query.collect())
        .and_then(|frame| Ok(frame.column("result")?.as_materialized_series().clone()));
    match (expected, actual) {
        (Ok(expected), Ok(actual)) => {
            assert_eq!(actual.dtype(), expected.dtype(), "{case}: dtype");
            let schema = query.unwrap().collect_schema().unwrap();
            assert_eq!(
                schema.get("result"),
                Some(expected.dtype()),
                "{case}: schema"
            );
            assert_eq!(actual.len(), expected.len(), "{case}: length");
            for row in 0..expected.len() {
                let (actual, expected) = (actual.get(row).unwrap(), expected.get(row).unwrap());
                assert!(
                    same(&actual, &expected),
                    "{case}: row {row}: foldv returned {actual}, Polars returned {expected}"
                );
            }
        },
        (Err(expected), Err(actual)) => assert_eq!(
            std::mem::discriminant(root_error(&actual)),
            std::mem::discriminant(root_error(&expected)),
            "{case}: foldv raised `{actual}`, Polars raised `{expected}`"
        ),
        (Err(expected), Ok(actual)) => {
            panic!("{case}: Polars raised `{expected}`, but foldv returned {actual:?}")
        },
        (Ok(expected), Err(actual)) => {
            panic!("{case}: foldv raised `{actual}`, but Polars returned {expected:?}")
        },
    }
}

fn ints() -> DataFrame {
    df!("l" => [Some(1_i64), None, Some(3), None, Some(-5)]).unwrap()
}

fn floats(dtype: DataType) -> DataFrame {
    df!("f" => [Some(1.0), Some(NAN), None, Some(-0.0), Some(0.0), Some(INF), Some(-INF), Some(NAN)])
        .unwrap()
        .lazy()
        .select([col("f").cast(dtype)])
        .collect()
        .unwrap()
}

fn bools() -> DataFrame {
    df!(
        "a" => [Some(true), Some(true), Some(true), Some(false), Some(false), Some(false), None, None, None],
        "b" => [Some(true), Some(false), None, Some(true), Some(false), None, Some(true), Some(false), None],
    )
    .unwrap()
}

fn i64_state(value: Option<i64>) -> Expr {
    lit(value.map_or(Scalar::null(DataType::Int64), |value| {
        Scalar::new(DataType::Int64, value.into())
    }))
}

const ARITHMETIC: [OperatorCase; 3] = [
    ("add", |left, right| left + right),
    ("subtract", |left, right| left - right),
    ("multiply", |left, right| left * right),
];

const COMPARISONS: [OperatorCase; 8] = [
    ("eq", Expr::eq),
    ("neq", Expr::neq),
    ("lt", Expr::lt),
    ("lt_eq", Expr::lt_eq),
    ("gt", Expr::gt),
    ("gt_eq", Expr::gt_eq),
    ("eq_missing", Expr::eq_missing),
    ("neq_missing", Expr::neq_missing),
];

#[test]
fn foldv_arithmetic_matches_polars() {
    let inputs = [
        ("int64", ints(), "l", i64_state(Some(1))),
        ("int64 null state", ints(), "l", i64_state(None)),
        ("float64", floats(DataType::Float64), "f", lit(1.0_f64)),
        (
            "float32",
            floats(DataType::Float32),
            "f",
            lit(1.0_f64).cast(DataType::Float32),
        ),
    ];
    for (op_name, op) in ARITHMETIC {
        for (input_name, df, column, initial_state) in &inputs {
            let body = |state: Expr| op(state, col(*column));
            let case = format!("{input_name} {op_name}");
            assert_matches_polars(&case, df, &body, initial_state.clone());
        }
    }
}

#[test]
fn foldv_float_comparisons_match_polars() {
    for dtype in [DataType::Float32, DataType::Float64] {
        let df = floats(dtype.clone());
        for (op_name, op) in COMPARISONS {
            for rhs in [Some(NAN), Some(0.0), Some(-0.0), Some(1.0), None] {
                let rhs_expr = match rhs {
                    Some(value) => lit(value).cast(dtype.clone()),
                    None => lit(NULL).cast(dtype.clone()),
                };
                let predicate = op(col("f"), rhs_expr);
                let case = format!("{dtype} {op_name} {rhs:?}");

                // Comparisons drive a branch and are also stored as a nullable Boolean state.
                let branch = |state: Expr| {
                    when(predicate.clone())
                        .then(state.clone() + lit(1_i64))
                        .otherwise(state)
                };
                assert_matches_polars(&case, &df, &branch, i64_state(Some(0)));
                let stored = |_state: Expr| predicate.clone();
                assert_matches_polars(&case, &df, &stored, lit(true));
            }
        }
    }
}

#[test]
fn foldv_integer_comparisons_match_polars() {
    for (op_name, op) in COMPARISONS {
        let predicate = op(col("l"), lit(3_i64));
        let branch = |state: Expr| {
            when(predicate.clone())
                .then(state.clone() + lit(1_i64))
                .otherwise(state)
        };
        assert_matches_polars(op_name, &ints(), &branch, i64_state(Some(0)));
        let stored = |_state: Expr| predicate.clone();
        assert_matches_polars(op_name, &ints(), &stored, lit(true));
    }
}

#[test]
fn foldv_boolean_operators_match_polars_kleene_logic() {
    let bodies: [BodyCase; 5] = [
        ("and", |_| col("a").and(col("b"))),
        ("or", |_| col("a").or(col("b"))),
        ("not", |_| col("a").not()),
        ("state and", |state| state.and(col("b"))),
        ("state or", |state| state.or(col("a"))),
    ];
    for (name, body) in bodies {
        for initial in [Some(true), Some(false), None] {
            let initial_state = match initial {
                Some(value) => lit(value),
                None => lit(NULL).cast(DataType::Boolean),
            };
            let case = format!("{name} from {initial:?}");
            assert_matches_polars(&case, &bools(), &body, initial_state);
        }
    }
}

#[test]
fn foldv_state_transitions_match_polars() {
    let bodies: [BodyCase; 7] = [
        ("recover", |state| {
            when(state.clone().is_null())
                .then(col("l"))
                .otherwise(state + col("l"))
        }),
        ("is_not_null", |state| {
            when(col("l").is_not_null())
                .then(state.clone() + lit(1_i64))
                .otherwise(state)
        }),
        ("null then", |_| {
            when(col("l").gt(lit(2_i64)))
                .then(lit(NULL))
                .otherwise(col("l"))
        }),
        ("null otherwise", |_| {
            when(col("l").gt(lit(0_i64)))
                .then(col("l"))
                .otherwise(lit(NULL))
        }),
        ("nested", |state| {
            when(col("l").gt(lit(0_i64)))
                .then(
                    when(state.clone().is_null())
                        .then(col("l"))
                        .otherwise(lit(NULL)),
                )
                .otherwise(state)
        }),
        ("fill_null", |state| {
            state.fill_null(col("l")) + col("l").fill_null(lit(0_i64))
        }),
        ("untyped null is_null", |state| {
            when(lit(NULL).is_null())
                .then(state.clone() + col("l"))
                .otherwise(state)
        }),
    ];
    for (name, body) in bodies {
        for initial in [Some(0), None] {
            let case = format!("{name} from {initial:?}");
            assert_matches_polars(&case, &ints(), &body, i64_state(initial));
        }
    }
}

#[test]
fn foldv_casts_match_polars() {
    let df = df!(
        "u64" => [Some(0_u64), Some(i64::MAX as u64), Some(1 << 63), Some(u64::MAX), None],
        "u32" => [Some(0_u32), Some(i32::MAX as u32), Some(1 << 31), Some(u32::MAX), None],
        "i64" => [Some(0_i64), Some(-1), Some(1 << 31), Some(-(1 << 31) - 1), None],
        "f64" => [Some(-0.5), Some(-2147483648.5), Some(2147483647.9), Some(NAN), None],
        "f64b" => [Some(-1.0), Some(4294967295.5), Some(INF), Some(-INF), Some(1e300)],
        "bool" => [Some(true), Some(false), None, Some(true), Some(false)],
    )
    .unwrap();
    let pairs = [
        ("u64", DataType::Int64),
        ("u64", DataType::Int32),
        ("u64", DataType::UInt32),
        ("u32", DataType::Int32),
        ("u32", DataType::Int64),
        ("i64", DataType::Int32),
        ("i64", DataType::UInt32),
        ("i64", DataType::UInt64),
        ("i64", DataType::Boolean),
        ("f64", DataType::Int32),
        ("f64", DataType::UInt32),
        ("f64", DataType::Int64),
        ("f64", DataType::UInt64),
        ("f64", DataType::Float32),
        ("f64", DataType::Boolean),
        ("f64b", DataType::UInt32),
        ("f64b", DataType::Int64),
        ("f64b", DataType::UInt64),
        ("f64b", DataType::Float32),
        ("bool", DataType::Int64),
        ("bool", DataType::Float64),
    ];
    let modes = [
        ("strict", CastOptions::Strict),
        ("non-strict", CastOptions::NonStrict),
        ("wrapping", CastOptions::Overflowing),
    ];
    for (source, target) in pairs {
        for (mode, options) in modes {
            let body = |_state: Expr| col(source).cast_with_options(target.clone(), options);
            let initial_state = if target == DataType::Boolean {
                lit(false)
            } else {
                lit(0).cast(target.clone())
            };
            let case = format!("{source} to {target} {mode}");
            assert_matches_polars(&case, &df, &body, initial_state);
        }
    }
}

#[test]
fn foldv_matches_polars_for_sliced_and_chunked_nullable_inputs() {
    let values = (0..40_i64)
        .map(|i| (i % 3 != 0).then_some(i))
        .collect::<Vec<_>>();
    let values = df!("l" => values).unwrap();
    let mut chunked = values.slice(0, 7);
    chunked.vstack_mut(&values.slice(7, 13)).unwrap();
    chunked.vstack_mut(&values.slice(20, 20)).unwrap();
    assert!(chunked.first_col_n_chunks() > 1);
    let flags = (0..40)
        .map(|i| (i % 5 != 0).then_some(i % 2 == 0))
        .collect::<Vec<_>>();
    let flags = df!("b" => flags).unwrap();

    let skip_nulls = |state: Expr| {
        when(col("l").is_null())
            .then(state.clone())
            .otherwise(state + col("l"))
    };
    let count_true = |state: Expr| {
        when(col("b"))
            .then(state.clone() + lit(1_i64))
            .otherwise(state)
    };
    let initial_state = i64_state(Some(0));
    assert_matches_polars(
        "sliced",
        &values.slice(5, 20),
        &skip_nulls,
        initial_state.clone(),
    );
    assert_matches_polars("chunked", &chunked, &skip_nulls, initial_state.clone());
    assert_matches_polars(
        "sliced bool",
        &flags.slice(3, 30),
        &count_true,
        initial_state,
    );
}

#[test]
fn foldv_infers_an_untyped_null_initial_state_like_polars() {
    let bodies: [InferenceCase; 4] = [
        (
            "recover",
            ints(),
            |state| {
                when(state.clone().is_null())
                    .then(col("l"))
                    .otherwise(state + col("l"))
            },
            DataType::Int64,
        ),
        (
            "float",
            floats(DataType::Float64),
            |state| state + col("f"),
            DataType::Float64,
        ),
        (
            "boolean",
            bools(),
            |state| state.or(col("a")),
            DataType::Boolean,
        ),
        (
            "fill_null",
            ints(),
            |state| state.fill_null(col("l")) * lit(2_i64),
            DataType::Int64,
        ),
    ];
    for (name, df, body, dtype) in bodies {
        // Polars gives `body(lit(NULL))` the same dtype that foldv must infer for the state.
        let native = df
            .clone()
            .lazy()
            .select([body(lit(NULL)).alias("native")])
            .collect_schema()
            .unwrap();
        assert_eq!(native.get("native"), Some(&dtype), "{name}: native dtype");
        assert_matches_polars_with(name, &df, &body, lit(NULL), lit(NULL).cast(dtype));
    }
}

#[test]
fn foldv_untyped_null_literals_match_polars() {
    // An untyped null takes the type of whatever it is combined with, as in Polars.
    let int_bodies: [BodyCase; 12] = [
        ("null + column", |_| lit(NULL) + col("l")),
        ("state * null + column", |state| {
            state * lit(NULL) + col("l")
        }),
        ("column > null", |state| {
            when(col("l").gt(lit(NULL)))
                .then(state.clone() + lit(1_i64))
                .otherwise(state)
        }),
        ("null == column", |state| {
            when(lit(NULL).eq(col("l")))
                .then(state.clone() + lit(1_i64))
                .otherwise(state)
        }),
        ("eq_missing null", |state| {
            when(col("l").eq_missing(lit(NULL)))
                .then(state.clone() + lit(1_i64))
                .otherwise(state)
        }),
        ("neq_missing null", |state| {
            when(col("l").neq_missing(lit(NULL)))
                .then(state.clone() + lit(1_i64))
                .otherwise(state)
        }),
        ("null cast", |_| lit(NULL).cast(DataType::Int64) + col("l")),
        ("null is_not_null", |state| {
            when(lit(NULL).is_not_null())
                .then(state.clone())
                .otherwise(state + col("l"))
        }),
        ("null then, null otherwise", |_| {
            when(col("l").gt(lit(0_i64)))
                .then(lit(NULL))
                .otherwise(lit(NULL))
        }),
        ("fill_null with null", |state| {
            state.fill_null(lit(NULL)) + col("l")
        }),
        ("null fill_null", |state| {
            lit(NULL).fill_null(col("l")) + state
        }),
        ("null then, state otherwise", |state| {
            when(col("l").is_null()).then(lit(NULL)).otherwise(state)
        }),
    ];
    for (name, body) in int_bodies {
        for initial in [Some(0), None] {
            let case = format!("{name} from {initial:?}");
            assert_matches_polars(&case, &ints(), &body, i64_state(initial));
        }
    }

    let float_eq_missing = |state: Expr| {
        when(col("f").eq_missing(lit(NULL)))
            .then(state.clone() + lit(1_i64))
            .otherwise(state)
    };
    assert_matches_polars(
        "float eq_missing null",
        &floats(DataType::Float64),
        &float_eq_missing,
        i64_state(Some(0)),
    );
    let stored = |_state: Expr| col("l").eq(lit(NULL));
    assert_matches_polars(
        "Boolean state = column == null",
        &ints(),
        &stored,
        lit(true),
    );
    let and_null = |_state: Expr| col("a").and(lit(NULL));
    assert_matches_polars("column and null", &bools(), &and_null, lit(true));
    let null_or = |_state: Expr| lit(NULL).or(col("a"));
    assert_matches_polars("null or column", &bools(), &null_or, lit(true));
}

#[test]
fn foldv_nan_checks_match_polars() {
    let checks: [BodyCase; 4] = [
        ("is_nan", Expr::is_nan),
        ("is_not_nan", Expr::is_not_nan),
        ("is_finite", Expr::is_finite),
        ("is_infinite", Expr::is_infinite),
    ];
    let inputs = [
        ("float64", floats(DataType::Float64), "f"),
        ("float32", floats(DataType::Float32), "f"),
        ("int64", ints(), "l"),
    ];
    for (check_name, check) in checks {
        for (input_name, df, column) in &inputs {
            let case = format!("{input_name} {check_name}");
            // Checks drive a branch and are also stored as a nullable Boolean state.
            let branch = |state: Expr| {
                when(check(col(*column)))
                    .then(state.clone() + lit(1_i64))
                    .otherwise(state)
            };
            assert_matches_polars(&case, df, &branch, i64_state(Some(0)));
            let stored = |_state: Expr| check(col(*column));
            assert_matches_polars(&case, df, &stored, lit(true));
        }
        // Polars rejects NaN checks on Booleans, and so must foldv.
        let boolean = |state: Expr| {
            when(check(col("a")))
                .then(state.clone() + lit(1_i64))
                .otherwise(state)
        };
        let case = format!("Boolean {check_name}");
        assert_matches_polars(&case, &bools(), &boolean, i64_state(Some(0)));
    }
}

#[test]
fn foldv_fill_nan_matches_polars() {
    // `fill_nan` keeps nulls; an integer filled with a float becomes Float64, as in Polars.
    let float64 = |state: Expr| state + col("f").fill_nan(lit(0.0_f64));
    assert_matches_polars(
        "float64",
        &floats(DataType::Float64),
        &float64,
        lit(0.0_f64),
    );
    let float32 = |state: Expr| state + col("f").fill_nan(lit(0.0_f32));
    let float32_state = lit(0.0_f64).cast(DataType::Float32);
    assert_matches_polars(
        "float32",
        &floats(DataType::Float32),
        &float32,
        float32_state,
    );
    let null_fill = |state: Expr| {
        when(state.clone().is_null())
            .then(col("f").fill_nan(lit(NULL)))
            .otherwise(state)
    };
    let null_state = lit(NULL).cast(DataType::Float64);
    assert_matches_polars(
        "float64 with null",
        &floats(DataType::Float64),
        &null_fill,
        null_state,
    );
    let integer = |state: Expr| state + col("l").fill_nan(lit(0.0_f64));
    assert_matches_polars("int64", &ints(), &integer, lit(0.0_f64));
    let integer_null = |state: Expr| state + col("l").fill_nan(lit(NULL));
    assert_matches_polars(
        "int64 with null",
        &ints(),
        &integer_null,
        i64_state(Some(0)),
    );
}

/// An untyped integer literal, like Python's `pl.lit(5)`; Rust's `lit(5)` is typed.
fn untyped_int(value: i128) -> Expr {
    Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(value)))
}

/// An untyped float literal, like Python's `pl.lit(1.5)`.
fn untyped_float(value: f64) -> Expr {
    Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Float(value)))
}

#[test]
fn foldv_literal_casts_match_polars() {
    // A cast of a literal applies the cast's overflow rule to the literal's own value. Values
    // beyond Int64 need Polars' Int128 support, which this test build does not enable.
    let literals = [
        ("5e9", untyped_int(5_000_000_000)),
        ("-1", untyped_int(-1)),
        ("1", untyped_int(1)),
        ("1.5", untyped_float(1.5)),
        ("NaN", untyped_float(f64::NAN)),
        ("typed 5e9", lit(5_000_000_000_i64)),
    ];
    let targets = [
        DataType::Int32,
        DataType::Int64,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float32,
        DataType::Boolean,
    ];
    let modes = [
        ("strict", CastOptions::Strict),
        ("non-strict", CastOptions::NonStrict),
        ("wrapping", CastOptions::Overflowing),
    ];
    for (literal_name, literal) in &literals {
        for target in &targets {
            for (mode, options) in modes {
                let cast = literal.clone().cast_with_options(target.clone(), options);
                let body =
                    |state: Expr| when(col("l").is_null()).then(state).otherwise(cast.clone());
                let initial_state = if *target == DataType::Boolean {
                    lit(false)
                } else {
                    lit(0).cast(target.clone())
                };
                let case = format!("{literal_name} to {target} {mode}");
                assert_matches_polars(&case, &ints(), &body, initial_state);
            }
        }
    }
}

#[test]
fn foldv_untyped_integer_literals_widen_like_polars() {
    // An untyped integer adopts the other operand's type only when its value fits there.
    // Values beyond Int64 need Polars' Int128 support, which this test build does not enable.
    let df = df!(
        "u32" => [Some(1_u32), None, Some(3)],
        "i32" => [Some(1_i32), Some(-2), None],
        "u64" => [Some(1_u64), Some(2), None],
        "f32" => [Some(1.5_f32), None, Some(2.5)],
    )
    .unwrap();
    let cases: [TypedBodyCase; 6] = [
        (
            "u32 + -1",
            |state| state + (col("u32") + untyped_int(-1)),
            DataType::Int64,
        ),
        (
            "i32 + 5e9",
            |state| state + (col("i32") + untyped_int(5_000_000_000)),
            DataType::Int64,
        ),
        (
            "u32 + 1",
            |state| state + (col("u32") + untyped_int(1)),
            DataType::UInt32,
        ),
        (
            "u64 + -1",
            |state| state + (col("u64") + untyped_int(-1)),
            DataType::Int64,
        ),
        (
            "f32 + 1",
            |state| state + (col("f32") + untyped_int(1)),
            DataType::Float32,
        ),
        (
            "u32 > -1",
            |state| {
                when(col("u32").gt(untyped_int(-1)))
                    .then(state.clone() + untyped_int(1))
                    .otherwise(state)
            },
            DataType::Int64,
        ),
    ];
    for (name, body, dtype) in cases {
        assert_matches_polars(name, &df, &body, lit(0).cast(dtype));
    }
}

#[test]
fn foldv_typed_null_literals_match_polars() {
    // A typed null keeps its own dtype instead of taking the other operand's.
    let df = df!("i32" => [Some(1_i32), Some(-1), None, Some(2)]).unwrap();
    let int64_null = || lit(NULL).cast(DataType::Int64);
    let typed_null = |dtype: DataType| lit(Scalar::null(dtype));
    let cases: [CapturingBodyCase; 5] = [
        (
            "typed null otherwise",
            Box::new(move |_| {
                when(col("i32").gt(lit(0_i32)))
                    .then(col("i32"))
                    .otherwise(typed_null(DataType::Int64))
            }),
        ),
        (
            "typed null in both branches",
            Box::new(move |_| {
                when(col("i32").gt(lit(0_i32)))
                    .then(typed_null(DataType::Int64))
                    .otherwise(typed_null(DataType::Int64))
            }),
        ),
        (
            "typed null plus column",
            Box::new(move |state| state + typed_null(DataType::Int64) + col("i32")),
        ),
        (
            "typed float null is_nan",
            Box::new(move |state| {
                when(typed_null(DataType::Float64).is_nan())
                    .then(state.clone())
                    .otherwise(state + col("i32"))
            }),
        ),
        (
            "typed null fill_null",
            Box::new(move |state| typed_null(DataType::Int64).fill_null(col("i32")) + state),
        ),
    ];
    for (name, body) in &cases {
        assert_matches_polars(name, &df, body.as_ref(), int64_null());
        assert_matches_polars(name, &df, body.as_ref(), lit(0).cast(DataType::Int64));
    }
}
