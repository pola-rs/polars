mod binary;

use std::borrow::Cow;

use polars_core::prelude::*;
use polars_core::utils::get_supertype;

use super::*;
use crate::dsl::function_expr::FunctionExpr;
use crate::logical_plan::optimizer::type_coercion::binary::process_binary;
use crate::logical_plan::Context;
use crate::utils::is_scan;

pub struct TypeCoercionRule {}

macro_rules! unpack {
    ($packed:expr) => {{
        match $packed {
            Some(payload) => payload,
            None => return Ok(None),
        }
    }};
}

// `dtype_other` comes from a column
// so we shrink literal so it fits into that column dtype.
fn shrink_literal(dtype_other: &DataType, literal: &LiteralValue) -> Option<DataType> {
    match (dtype_other, literal) {
        (DataType::UInt64, LiteralValue::Int64(v)) => {
            if *v > 0 {
                return Some(DataType::UInt64);
            }
        }
        (DataType::UInt64, LiteralValue::Int32(v)) => {
            if *v > 0 {
                return Some(DataType::UInt64);
            }
        }
        #[cfg(feature = "dtype-i16")]
        (DataType::UInt64, LiteralValue::Int16(v)) => {
            if *v > 0 {
                return Some(DataType::UInt64);
            }
        }
        #[cfg(feature = "dtype-i8")]
        (DataType::UInt64, LiteralValue::Int8(v)) => {
            if *v > 0 {
                return Some(DataType::UInt64);
            }
        }
        (DataType::UInt32, LiteralValue::Int64(v)) => {
            if *v > 0 && *v < u32::MAX as i64 {
                return Some(DataType::UInt32);
            }
        }
        (DataType::UInt32, LiteralValue::Int32(v)) => {
            if *v > 0 {
                return Some(DataType::UInt32);
            }
        }
        #[cfg(feature = "dtype-i16")]
        (DataType::UInt32, LiteralValue::Int16(v)) => {
            if *v > 0 {
                return Some(DataType::UInt32);
            }
        }
        #[cfg(feature = "dtype-i8")]
        (DataType::UInt32, LiteralValue::Int8(v)) => {
            if *v > 0 {
                return Some(DataType::UInt32);
            }
        }
        (DataType::UInt16, LiteralValue::Int64(v)) => {
            if *v > 0 && *v < u16::MAX as i64 {
                return Some(DataType::UInt16);
            }
        }
        (DataType::UInt16, LiteralValue::Int32(v)) => {
            if *v > 0 && *v < u16::MAX as i32 {
                return Some(DataType::UInt16);
            }
        }
        #[cfg(feature = "dtype-i16")]
        (DataType::UInt16, LiteralValue::Int16(v)) => {
            if *v > 0 {
                return Some(DataType::UInt16);
            }
        }
        #[cfg(feature = "dtype-i8")]
        (DataType::UInt16, LiteralValue::Int8(v)) => {
            if *v > 0 {
                return Some(DataType::UInt16);
            }
        }
        (DataType::UInt8, LiteralValue::Int64(v)) => {
            if *v > 0 && *v < u8::MAX as i64 {
                return Some(DataType::UInt8);
            }
        }
        (DataType::UInt8, LiteralValue::Int32(v)) => {
            if *v > 0 && *v < u8::MAX as i32 {
                return Some(DataType::UInt8);
            }
        }
        #[cfg(feature = "dtype-i16")]
        (DataType::UInt8, LiteralValue::Int16(v)) => {
            if *v > 0 && *v < u8::MAX as i16 {
                return Some(DataType::UInt8);
            }
        }
        #[cfg(feature = "dtype-i8")]
        (DataType::UInt8, LiteralValue::Int8(v)) => {
            if *v > 0 && *v < u8::MAX as i8 {
                return Some(DataType::UInt8);
            }
        }
        (DataType::Int32, LiteralValue::Int64(v)) => {
            if *v <= i32::MAX as i64 {
                return Some(DataType::Int32);
            }
        }
        (DataType::Int16, LiteralValue::Int64(v)) => {
            if *v <= i16::MAX as i64 {
                return Some(DataType::Int16);
            }
        }
        (DataType::Int16, LiteralValue::Int32(v)) => {
            if *v <= i16::MAX as i32 {
                return Some(DataType::Int16);
            }
        }
        (DataType::Int8, LiteralValue::Int64(v)) => {
            if *v <= i8::MAX as i64 {
                return Some(DataType::Int8);
            }
        }
        (DataType::Int8, LiteralValue::Int32(v)) => {
            if *v <= i8::MAX as i32 {
                return Some(DataType::Int8);
            }
        }
        #[cfg(feature = "dtype-i16")]
        (DataType::Int8, LiteralValue::Int16(v)) => {
            if *v <= i8::MAX as i16 {
                return Some(DataType::Int8);
            }
        }
        _ => {
            // the rest is done by supertypes.
        }
    }
    None
}

/// determine if we use the supertype or not. For instance when we have a column Int64 and we compare with literal UInt32
/// it would be wasteful to cast the column instead of the literal.
fn modify_supertype(
    mut st: DataType,
    left: &AExpr,
    right: &AExpr,
    type_left: &DataType,
    type_right: &DataType,
) -> DataType {
    // only interesting on numerical types
    // other types will always use the supertype.
    if type_left.is_numeric() && type_right.is_numeric() {
        use AExpr::*;
        match (left, right) {
            // don't let the literal f64 coerce the f32 column
            (
                Literal(LiteralValue::Float64(_) | LiteralValue::Int32(_) | LiteralValue::Int64(_)),
                _,
            ) if matches!(type_right, DataType::Float32) => st = DataType::Float32,
            (
                _,
                Literal(LiteralValue::Float64(_) | LiteralValue::Int32(_) | LiteralValue::Int64(_)),
            ) if matches!(type_left, DataType::Float32) => st = DataType::Float32,
            // always make sure that we cast to floats if one of the operands is float
            (Literal(lv), _) | (_, Literal(lv)) if lv.is_float() => {}

            // TODO: see if we can activate this for columns as well.
            // shrink the literal value if it fits in the column dtype
            (Literal(LiteralValue::Series(_)), Literal(lv)) => {
                if let Some(dtype) = shrink_literal(type_left, lv) {
                    st = dtype;
                }
            }
            // shrink the literal value if it fits in the column dtype
            (Literal(lv), Literal(LiteralValue::Series(_))) => {
                if let Some(dtype) = shrink_literal(type_right, lv) {
                    st = dtype;
                }
            }
            // do nothing and use supertype
            (Literal(_), Literal(_)) => {}

            // cast literal to right type if they fit in the range
            (Literal(value), _) => {
                if let Some(lit_val) = value.to_anyvalue() {
                    if type_right.value_within_range(lit_val) {
                        st = type_right.clone();
                    }
                }
            }
            // cast literal to left type
            (_, Literal(value)) => {
                if let Some(lit_val) = value.to_anyvalue() {
                    if type_left.value_within_range(lit_val) {
                        st = type_left.clone();
                    }
                }
            }
            // do nothing
            _ => {}
        }
    } else {
        use DataType::*;
        match (type_left, type_right, left, right) {
            // if the we compare a categorical to a literal string we want to cast the literal to categorical
            #[cfg(feature = "dtype-categorical")]
            (Categorical(_), Utf8, _, AExpr::Literal(_))
            | (Utf8, Categorical(_), AExpr::Literal(_), _) => {
                st = Categorical(None);
            }
            // when then expression literals can have a different list type.
            // so we cast the literal to the other hand side.
            (List(inner), List(other), _, AExpr::Literal(_))
            | (List(other), List(inner), AExpr::Literal(_), _)
                if inner != other =>
            {
                st = DataType::List(inner.clone())
            }
            // do nothing
            _ => {}
        }
    }
    st
}

fn get_input(lp_arena: &Arena<ALogicalPlan>, lp_node: Node) -> [Option<Node>; 2] {
    let plan = lp_arena.get(lp_node);
    let mut inputs = [None, None];

    // Used to get the schema of the input.
    if is_scan(plan) {
        inputs[0] = Some(lp_node);
    } else {
        plan.copy_inputs(&mut inputs);
    };
    inputs
}

fn get_schema(lp_arena: &Arena<ALogicalPlan>, lp_node: Node) -> Cow<'_, SchemaRef> {
    match get_input(lp_arena, lp_node) {
        [Some(input), _] => lp_arena.get(input).schema(lp_arena),
        // files don't have an input, so we must take their schema
        [None, _] => Cow::Borrowed(lp_arena.get(lp_node).scan_schema()),
    }
}

fn get_aexpr_and_type<'a>(
    expr_arena: &'a Arena<AExpr>,
    e: Node,
    input_schema: &Schema,
) -> Option<(&'a AExpr, DataType)> {
    let ae = expr_arena.get(e);
    Some((
        ae,
        ae.get_type(input_schema, Context::Default, expr_arena)
            .ok()?,
    ))
}

#[cfg(feature = "python")]
fn err_date_str_compare() -> PolarsResult<()> {
    Err(PolarsError::ComputeError(
        "Cannot compare 'date/datetime/time' to a string value.\n\
        Create native python {{ 'date', 'datetime', 'time' }} or compare to a temporal column."
            .into(),
    ))
}

#[cfg(not(feature = "python"))]
fn err_date_str_compare() -> PolarsResult<()> {
    Err(PolarsError::ComputeError(
        "Cannot compare 'date/datetime/time' to a string value.".into(),
    ))
}

impl OptimizationRule for TypeCoercionRule {
    fn optimize_expr(
        &self,
        expr_arena: &mut Arena<AExpr>,
        expr_node: Node,
        lp_arena: &Arena<ALogicalPlan>,
        lp_node: Node,
    ) -> PolarsResult<Option<AExpr>> {
        let expr = expr_arena.get(expr_node);
        let out = match *expr {
            AExpr::Ternary {
                truthy: truthy_node,
                falsy: falsy_node,
                predicate,
            } => {
                let input_schema = get_schema(lp_arena, lp_node);
                let (truthy, type_true) =
                    unpack!(get_aexpr_and_type(expr_arena, truthy_node, &input_schema));
                let (falsy, type_false) =
                    unpack!(get_aexpr_and_type(expr_arena, falsy_node, &input_schema));

                unpack!(early_escape(&type_true, &type_false));
                let st = unpack!(get_supertype(&type_true, &type_false));
                let st = modify_supertype(st, truthy, falsy, &type_true, &type_false);

                // only cast if the type is not already the super type.
                // this can prevent an expensive flattening and subsequent aggregation
                // in a groupby context. To be able to cast the groups need to be
                // flattened
                let new_node_truthy = if type_true != st {
                    expr_arena.add(AExpr::Cast {
                        expr: truthy_node,
                        data_type: st.clone(),
                        strict: false,
                    })
                } else {
                    truthy_node
                };

                let new_node_falsy = if type_false != st {
                    expr_arena.add(AExpr::Cast {
                        expr: falsy_node,
                        data_type: st,
                        strict: false,
                    })
                } else {
                    falsy_node
                };

                Some(AExpr::Ternary {
                    truthy: new_node_truthy,
                    falsy: new_node_falsy,
                    predicate,
                })
            }
            AExpr::BinaryExpr {
                left: node_left,
                op,
                right: node_right,
            } => return process_binary(expr_arena, lp_arena, lp_node, node_left, op, node_right),
            #[cfg(feature = "is_in")]
            AExpr::Function {
                function: FunctionExpr::IsIn,
                ref input,
                options,
            } => {
                let input_schema = get_schema(lp_arena, lp_node);
                let other_node = input[1];
                let (_, type_left) =
                    unpack!(get_aexpr_and_type(expr_arena, input[0], &input_schema));
                let (_, type_other) =
                    unpack!(get_aexpr_and_type(expr_arena, other_node, &input_schema));

                unpack!(early_escape(&type_left, &type_other));

                let casted_expr = match (&type_left, &type_other) {
                    // cast both local and global string cache
                    // note that there might not yet be a rev
                    #[cfg(feature = "dtype-categorical")]
                    (DataType::Categorical(_), DataType::Utf8) => {
                        AExpr::Cast {
                            expr: other_node,
                            data_type: DataType::Categorical(None),
                            // does not matter
                            strict: false,
                        }
                    }
                    (DataType::List(_), _) | (_, DataType::List(_)) => return Ok(None),
                    #[cfg(feature = "dtype-struct")]
                    (DataType::Struct(_), _) | (_, DataType::Struct(_)) => return Ok(None),
                    // if right is another type, we cast it to left
                    // we do not use super-type as an `is_in` operation should not
                    // cast the whole column implicitly.
                    (a, b) if a != b => {
                        AExpr::Cast {
                            expr: other_node,
                            data_type: type_left,
                            // does not matter
                            strict: false,
                        }
                    }
                    // types are equal, do nothing
                    _ => return Ok(None),
                };

                let mut input = input.clone();
                let other_input = expr_arena.add(casted_expr);
                input[1] = other_input;

                Some(AExpr::Function {
                    function: FunctionExpr::IsIn,
                    input,
                    options,
                })
            }
            // fill null has a supertype set during projection
            // to make the schema known before the optimization phase
            AExpr::Function {
                function: FunctionExpr::FillNull { ref super_type },
                ref input,
                options,
            } => {
                let input_schema = get_schema(lp_arena, lp_node);
                let other_node = input[1];
                let (left, type_left) =
                    unpack!(get_aexpr_and_type(expr_arena, input[0], &input_schema));
                let (fill_value, type_fill_value) =
                    unpack!(get_aexpr_and_type(expr_arena, other_node, &input_schema));

                let new_st = unpack!(get_supertype(&type_left, &type_fill_value));
                let new_st =
                    modify_supertype(new_st, left, fill_value, &type_left, &type_fill_value);
                if &new_st != super_type {
                    Some(AExpr::Function {
                        function: FunctionExpr::FillNull { super_type: new_st },
                        input: input.clone(),
                        options,
                    })
                } else {
                    None
                }
            }
            // generic type coercion of any function.
            AExpr::Function {
                // only for `DataType::Unknown` as it still has to be set.
                ref function,
                ref input,
                mut options,
            } if options.cast_to_supertypes => {
                // satisfy bchk
                let function = function.clone();
                let input = input.clone();

                let input_schema = get_schema(lp_arena, lp_node);
                let self_node = input[0];
                let (self_ae, type_self) =
                    unpack!(get_aexpr_and_type(expr_arena, self_node, &input_schema));

                let mut super_type = type_self.clone();
                for other in &input[1..] {
                    let (other, type_other) =
                        unpack!(get_aexpr_and_type(expr_arena, *other, &input_schema));

                    // early return until Unknown is set
                    unpack!(early_escape(&super_type, &type_other));
                    let new_st = unpack!(get_supertype(&super_type, &type_other));
                    super_type = modify_supertype(new_st, self_ae, other, &type_self, &type_other)
                }
                // only cast if the type is not already the super type.
                // this can prevent an expensive flattening and subsequent aggregation
                // in a groupby context. To be able to cast the groups need to be
                // flattened
                let new_node_self = if type_self != super_type {
                    expr_arena.add(AExpr::Cast {
                        expr: self_node,
                        data_type: super_type.clone(),
                        strict: false,
                    })
                } else {
                    self_node
                };
                let mut new_nodes = Vec::with_capacity(input.len());
                new_nodes.push(new_node_self);

                for other_node in &input[1..] {
                    let type_other =
                        match get_aexpr_and_type(expr_arena, *other_node, &input_schema) {
                            Some((_, type_other)) => type_other,
                            None => return Ok(None),
                        };
                    let new_node_other = if type_other != super_type {
                        expr_arena.add(AExpr::Cast {
                            expr: *other_node,
                            data_type: super_type.clone(),
                            strict: false,
                        })
                    } else {
                        *other_node
                    };

                    new_nodes.push(new_node_other)
                }
                // ensure we don't go through this on next iteration
                options.cast_to_supertypes = false;
                Some(AExpr::Function {
                    function,
                    input: new_nodes,
                    options,
                })
            }
            _ => None,
        };
        Ok(out)
    }
}

fn early_escape(type_self: &DataType, type_other: &DataType) -> Option<()> {
    if type_self == type_other
        || matches!(type_self, DataType::Unknown)
        || matches!(type_other, DataType::Unknown)
    {
        None
    } else {
        Some(())
    }
}

#[cfg(test)]
#[cfg(feature = "dtype-categorical")]
mod test {
    use polars_core::prelude::*;

    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_categorical_utf8() {
        let mut rules: Vec<Box<dyn OptimizationRule>> = vec![Box::new(TypeCoercionRule {})];
        let schema = Schema::from(vec![Field::new("fruits", DataType::Categorical(None))]);

        let expr = col("fruits").eq(lit("somestr"));
        let out = optimize_expr(expr.clone(), schema.clone(), &mut rules);
        // we test that the fruits column is not casted to utf8 for the comparison
        assert_eq!(out, expr);

        let expr = col("fruits") + (lit("somestr"));
        let out = optimize_expr(expr, schema, &mut rules);
        let expected = col("fruits").cast(DataType::Utf8) + lit("somestr");
        assert_eq!(out, expected);
    }
}
