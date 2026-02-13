use std::fmt::Write;

use polars_core::datatypes::AnyValue;
use polars_core::prelude::{TimeUnit, TimeZone};

use crate::prelude::*;

#[derive(Default, Copy, Clone)]
pub struct PyarrowArgs {
    // pyarrow doesn't allow `filter([True, False])`
    // but does allow `filter(field("a").isin([True, False]))`
    allow_literal_series: bool,
    /// When true, AND expressions with non-convertible children return only
    /// the convertible parts. The caller MUST ensure the full predicate is
    /// preserved elsewhere (e.g., as a SELECTION in the IR plan).
    partial_and: bool,
}

impl PyarrowArgs {
    pub fn with_partial_and(mut self) -> Self {
        self.partial_and = true;
        self
    }
}

fn to_py_datetime(v: i64, tu: &TimeUnit, tz: Option<&TimeZone>) -> String {
    // note: `to_py_datetime` and the `Datetime`
    // dtype have to be in-scope on the python side
    match tz {
        None => format!("to_py_datetime({},'{}')", v, tu.to_ascii()),
        Some(tz) => format!("to_py_datetime({},'{}','{}')", v, tu.to_ascii(), tz),
    }
}

fn sanitize(name: &str) -> Option<&str> {
    if name.chars().all(|c| match c {
        ' ' => true,
        '-' => true,
        '_' => true,
        c => c.is_alphanumeric(),
    }) {
        Some(name)
    } else {
        None
    }
}

/// Format a `Series` as a Python list literal, e.g. `['a','b']` or `[1,2,3]`.
/// Returns `None` if the series is empty, too large (>100), or contains
/// values that cannot be safely represented.
fn series_to_pa_list(s: &polars_core::prelude::Series) -> Option<String> {
    if s.is_empty() || s.len() > 100 {
        return None;
    }
    let mut list_repr = String::with_capacity(s.len() * 5);
    list_repr.push('[');
    for av in s.iter() {
        match av {
            AnyValue::Boolean(v) => {
                let s = if v { "True" } else { "False" };
                write!(list_repr, "{s},").unwrap();
            },
            #[cfg(feature = "dtype-datetime")]
            AnyValue::Datetime(v, tu, tz) => {
                let dtm = to_py_datetime(v, &tu, tz);
                write!(list_repr, "{dtm},").unwrap();
            },
            #[cfg(feature = "dtype-date")]
            AnyValue::Date(v) => {
                write!(list_repr, "to_py_date({v}),").unwrap();
            },
            AnyValue::String(s) => {
                let _ = sanitize(s)?;
                write!(list_repr, "{av},").unwrap();
            },
            // Hard to sanitize
            AnyValue::Binary(_) | AnyValue::List(_) => return None,
            #[cfg(feature = "dtype-array")]
            AnyValue::Array(_, _) => return None,
            #[cfg(feature = "dtype-struct")]
            AnyValue::Struct(_, _, _) => return None,
            _ => {
                write!(list_repr, "{av},").unwrap();
            },
        }
    }
    // pop last comma
    list_repr.pop();
    list_repr.push(']');
    Some(list_repr)
}

// convert to a pyarrow expression that can be evaluated with pythons eval
pub fn predicate_to_pa(
    predicate: Node,
    expr_arena: &Arena<AExpr>,
    args: PyarrowArgs,
) -> Option<String> {
    match expr_arena.get(predicate) {
        AExpr::BinaryExpr { left, right, op } => {
            if op.is_comparison_or_bitwise() {
                if args.partial_and && matches!(op, Operator::And) {
                    // AND: allow partial conversion. Dropped parts remain as
                    // post-scan SELECTION filters in the Polars engine.
                    let left_result = predicate_to_pa(*left, expr_arena, args);
                    let right_result = predicate_to_pa(*right, expr_arena, args);
                    match (left_result, right_result) {
                        (Some(l), Some(r)) => Some(format!("({l} & {r})")),
                        (Some(l), None) => Some(l),
                        (None, Some(r)) => Some(r),
                        (None, None) => None,
                    }
                } else {
                    let left = predicate_to_pa(*left, expr_arena, args)?;
                    let right = predicate_to_pa(*right, expr_arena, args)?;
                    Some(format!("({left} {op} {right})"))
                }
            } else {
                None
            }
        },
        AExpr::Column(name) => {
            let name = sanitize(name)?;
            Some(format!("pa.compute.field('{name}')"))
        },
        AExpr::Literal(LiteralValue::Series(s)) => {
            if !args.allow_literal_series {
                None
            } else {
                series_to_pa_list(s)
            }
        },
        // Guard against DynLiteralValue::List reaching to_any_value() which
        // contains a todo!() panic. In practice the optimizer materializes
        // Dyn(List) to Scalar before predicate_to_pa is called.
        AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::List(_))) => None,
        AExpr::Literal(lv) => {
            let av = lv.to_any_value()?;
            let dtype = av.dtype();
            match av.as_borrowed() {
                AnyValue::String(s) => {
                    let s = sanitize(s)?;
                    Some(format!("'{s}'"))
                },
                AnyValue::Boolean(val) => {
                    // python bools are capitalized
                    if val {
                        Some("pa.compute.scalar(True)".to_string())
                    } else {
                        Some("pa.compute.scalar(False)".to_string())
                    }
                },
                #[cfg(feature = "dtype-date")]
                AnyValue::Date(v) => {
                    // the function `to_py_date` and the `Date`
                    // dtype have to be in scope on the python side
                    Some(format!("to_py_date({v})"))
                },
                #[cfg(feature = "dtype-datetime")]
                AnyValue::Datetime(v, tu, tz) => Some(to_py_datetime(v, &tu, tz)),
                AnyValue::List(s) => {
                    // List values appear here when a Scalar literal wraps a
                    // list-typed AnyValue (e.g. `is_in` values after optimizer
                    // materializes DynLiteralValue::List → Scalar).
                    if args.allow_literal_series {
                        series_to_pa_list(&s)
                    } else {
                        None
                    }
                },
                // Hard to sanitize
                AnyValue::Binary(_) => None,
                #[cfg(feature = "dtype-array")]
                AnyValue::Array(_, _) => None,
                #[cfg(feature = "dtype-struct")]
                AnyValue::Struct(_, _, _) => None,
                // Activate once pyarrow supports them
                // #[cfg(feature = "dtype-time")]
                // AnyValue::Time(v) => {
                //     // the function `to_py_time` has to be in scope
                //     // on the python side
                //     Some(format!("to_py_time(value={v})"))
                // }
                // #[cfg(feature = "dtype-duration")]
                // AnyValue::Duration(v, tu) => {
                //     // the function `to_py_timedelta` has to be in scope
                //     // on the python side
                //     Some(format!(
                //         "to_py_timedelta(value={}, tu='{}')",
                //         v,
                //         tu.to_ascii()
                //     ))
                // }
                av => {
                    if dtype.is_float() {
                        let val = av.extract::<f64>()?;
                        Some(format!("{val}"))
                    } else if dtype.is_integer() {
                        let val = av.extract::<i64>()?;
                        Some(format!("{val}"))
                    } else {
                        None
                    }
                },
            }
        },
        #[cfg(feature = "is_in")]
        AExpr::Function {
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { .. }),
            input,
            ..
        } => {
            let col = predicate_to_pa(input.first()?.node(), expr_arena, args)?;
            let mut args = args;
            args.allow_literal_series = true;
            let values = predicate_to_pa(input.get(1)?.node(), expr_arena, args)?;

            Some(format!("({col}).isin({values})"))
        },
        #[cfg(feature = "is_between")]
        AExpr::Function {
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsBetween { closed }),
            input,
            ..
        } => {
            if !matches!(expr_arena.get(input.first()?.node()), AExpr::Column(_)) {
                None
            } else {
                let col = predicate_to_pa(input.first()?.node(), expr_arena, args)?;
                let left_cmp_op = match closed {
                    ClosedInterval::None | ClosedInterval::Right => Operator::Gt,
                    ClosedInterval::Both | ClosedInterval::Left => Operator::GtEq,
                };
                let right_cmp_op = match closed {
                    ClosedInterval::None | ClosedInterval::Left => Operator::Lt,
                    ClosedInterval::Both | ClosedInterval::Right => Operator::LtEq,
                };

                let lower = predicate_to_pa(input.get(1)?.node(), expr_arena, args)?;
                let upper = predicate_to_pa(input.get(2)?.node(), expr_arena, args)?;

                Some(format!(
                    "(({col} {left_cmp_op} {lower}) & ({col} {right_cmp_op} {upper}))"
                ))
            }
        },
        AExpr::Function {
            function, input, ..
        } => {
            let input = input.first().unwrap().node();
            let input = predicate_to_pa(input, expr_arena, args)?;

            match function {
                IRFunctionExpr::Boolean(IRBooleanFunction::Not) => Some(format!("~({input})")),
                IRFunctionExpr::Boolean(IRBooleanFunction::IsNull) => {
                    Some(format!("({input}).is_null()"))
                },
                IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNull) => {
                    Some(format!("~({input}).is_null()"))
                },
                _ => None,
            }
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::{NamedFrom, Scalar, Series};
    use polars_utils::pl_str::PlSmallStr;

    use super::*;
    use crate::plans::expr_ir::{ExprIR, OutputName};

    /// Helper: add a column reference node to the arena.
    fn col(arena: &mut Arena<AExpr>, name: &str) -> Node {
        arena.add(AExpr::Column(PlSmallStr::from(name)))
    }

    /// Helper: add an integer literal node to the arena.
    fn lit_int(arena: &mut Arena<AExpr>, v: i128) -> Node {
        arena.add(AExpr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(v))))
    }

    /// Helper: add a binary expression node to the arena.
    fn binexpr(arena: &mut Arena<AExpr>, left: Node, op: Operator, right: Node) -> Node {
        arena.add(AExpr::BinaryExpr { left, op, right })
    }

    /// Build `col(name) > lit(v)` — a simple convertible comparison.
    fn col_gt_lit(arena: &mut Arena<AExpr>, name: &str, v: i128) -> Node {
        let c = col(arena, name);
        let l = lit_int(arena, v);
        binexpr(arena, c, Operator::Gt, l)
    }

    /// Build `col(name) < lit(v)` — a simple convertible comparison.
    fn col_lt_lit(arena: &mut Arena<AExpr>, name: &str, v: i128) -> Node {
        let c = col(arena, name);
        let l = lit_int(arena, v);
        binexpr(arena, c, Operator::Lt, l)
    }

    /// Build `col(a) * col(b) > lit(v)` — non-convertible (Multiply is not
    /// comparison/bitwise).
    fn mul_gt_lit(arena: &mut Arena<AExpr>, a: &str, b: &str, v: i128) -> Node {
        let ca = col(arena, a);
        let cb = col(arena, b);
        let mul = binexpr(arena, ca, Operator::Multiply, cb);
        let l = lit_int(arena, v);
        binexpr(arena, mul, Operator::Gt, l)
    }

    /// Build an `is_in` AExpr::Function node: `col(name).is_in(values_literal)`.
    #[cfg(feature = "is_in")]
    fn is_in_node(arena: &mut Arena<AExpr>, col_name: &str, values_node: Node) -> Node {
        let col_node = col(arena, col_name);
        let col_ir = ExprIR::new(col_node, OutputName::ColumnLhs(PlSmallStr::from(col_name)));
        let val_ir = ExprIR::from_node(values_node, arena);
        arena.add(AExpr::Function {
            input: vec![col_ir, val_ir],
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { nulls_equal: false }),
            options: FunctionOptions::default(),
        })
    }

    /// Build a Scalar literal node containing a list of strings.
    #[cfg(feature = "is_in")]
    fn lit_str_list(arena: &mut Arena<AExpr>, values: &[&str]) -> Node {
        let series = Series::new(PlSmallStr::from("values"), values);
        let scalar = Scalar::new_list(series);
        arena.add(AExpr::Literal(LiteralValue::Scalar(scalar)))
    }

    /// Build a Scalar literal node containing a list of integers.
    #[cfg(feature = "is_in")]
    fn lit_int_list(arena: &mut Arena<AExpr>, values: &[i64]) -> Node {
        let series = Series::new(PlSmallStr::from("values"), values);
        let scalar = Scalar::new_list(series);
        arena.add(AExpr::Literal(LiteralValue::Scalar(scalar)))
    }

    #[test]
    fn test_full_and_converts_with_partial() {
        let mut arena = Arena::new();
        let left = col_gt_lit(&mut arena, "a", 5);
        let right = col_lt_lit(&mut arena, "b", 10);
        let and = binexpr(&mut arena, left, Operator::And, right);

        let args = PyarrowArgs::default().with_partial_and();
        let result = predicate_to_pa(and, &arena, args);
        assert!(result.is_some());
        let s = result.unwrap();
        assert!(s.contains('&'), "expected AND: {s}");
        assert!(s.contains("'a'"), "expected col a: {s}");
        assert!(s.contains("'b'"), "expected col b: {s}");
    }

    #[test]
    fn test_mixed_and_partial_on() {
        let mut arena = Arena::new();
        let convertible = col_gt_lit(&mut arena, "a", 5);
        let non_convertible = mul_gt_lit(&mut arena, "a", "b", 10);
        let and = binexpr(&mut arena, convertible, Operator::And, non_convertible);

        let args = PyarrowArgs::default().with_partial_and();
        let result = predicate_to_pa(and, &arena, args);
        assert!(
            result.is_some(),
            "partial_and should return convertible side"
        );
        let s = result.unwrap();
        assert!(s.contains("'a'"), "expected col a: {s}");
        assert!(s.contains("5"), "expected literal 5: {s}");
    }

    #[test]
    fn test_mixed_and_partial_off() {
        let mut arena = Arena::new();
        let convertible = col_gt_lit(&mut arena, "a", 5);
        let non_convertible = mul_gt_lit(&mut arena, "a", "b", 10);
        let and = binexpr(&mut arena, convertible, Operator::And, non_convertible);

        let result = predicate_to_pa(and, &arena, Default::default());
        assert!(
            result.is_none(),
            "without partial_and, mixed AND should be None"
        );
    }

    #[test]
    fn test_all_non_convertible_and() {
        let mut arena = Arena::new();
        let left = mul_gt_lit(&mut arena, "a", "b", 5);
        let right = mul_gt_lit(&mut arena, "c", "d", 10);
        let and = binexpr(&mut arena, left, Operator::And, right);

        let args = PyarrowArgs::default().with_partial_and();
        let result = predicate_to_pa(and, &arena, args);
        assert!(result.is_none(), "all non-convertible AND should be None");
    }

    #[test]
    fn test_or_with_non_convertible_never_splits() {
        let mut arena = Arena::new();
        let convertible = col_gt_lit(&mut arena, "a", 5);
        let non_convertible = mul_gt_lit(&mut arena, "a", "b", 10);
        let or = binexpr(&mut arena, convertible, Operator::Or, non_convertible);

        let args = PyarrowArgs::default().with_partial_and();
        let result = predicate_to_pa(or, &arena, args);
        assert!(
            result.is_none(),
            "OR with non-convertible side must be None"
        );
    }

    #[test]
    fn test_nested_and_one_bad_leaf() {
        let mut arena = Arena::new();
        // ((a > 5) & (b < 10)) & (c * d > 0)
        let left_inner = col_gt_lit(&mut arena, "a", 5);
        let right_inner = col_lt_lit(&mut arena, "b", 10);
        let inner_and = binexpr(&mut arena, left_inner, Operator::And, right_inner);
        let bad_leaf = mul_gt_lit(&mut arena, "c", "d", 0);
        let outer_and = binexpr(&mut arena, inner_and, Operator::And, bad_leaf);

        let args = PyarrowArgs::default().with_partial_and();
        let result = predicate_to_pa(outer_and, &arena, args);
        assert!(
            result.is_some(),
            "nested AND should return convertible branch"
        );
        let s = result.unwrap();
        assert!(s.contains("'a'"), "expected col a: {s}");
        assert!(s.contains("'b'"), "expected col b: {s}");
        assert!(
            !s.contains("'c'"),
            "should not contain non-convertible col c: {s}"
        );
        assert!(
            !s.contains("'d'"),
            "should not contain non-convertible col d: {s}"
        );
    }

    #[cfg(feature = "is_in")]
    #[test]
    fn test_is_in_string_list_converts() {
        let mut arena = Arena::new();
        let vals = lit_str_list(&mut arena, &["Engineering", "HR"]);
        let is_in = is_in_node(&mut arena, "department", vals);

        let result = predicate_to_pa(is_in, &arena, Default::default());
        assert!(result.is_some(), "is_in with string list should convert");
        let s = result.unwrap();
        assert!(s.contains("isin"), "expected isin: {s}");
        assert!(s.contains("'department'"), "expected column name: {s}");
        assert!(
            s.contains("Engineering") && s.contains("HR"),
            "expected list values: {s}"
        );
    }

    #[cfg(feature = "is_in")]
    #[test]
    fn test_is_in_int_list_converts() {
        let mut arena = Arena::new();
        let vals = lit_int_list(&mut arena, &[1, 2, 3]);
        let is_in = is_in_node(&mut arena, "id", vals);

        let result = predicate_to_pa(is_in, &arena, Default::default());
        assert!(result.is_some(), "is_in with int list should convert");
        let s = result.unwrap();
        assert!(s.contains("isin"), "expected isin: {s}");
        assert!(s.contains("[1,2,3]"), "expected int list: {s}");
    }
}
