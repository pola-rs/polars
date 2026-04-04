use std::fmt::Write;

use polars_core::datatypes::AnyValue;
use polars_core::prelude::DataType;

use crate::prelude::*;

const TAG_FIELD: &str = "field";
const TAG_BINOP: &str = "binop";
const TAG_IS_NULL: &str = "is_null";
const TAG_IS_NOT_NULL: &str = "is_not_null";
const TAG_LIT_STR: &str = "lit_str";
const TAG_LIT_BOOL: &str = "lit_bool";
const TAG_LIT_I64: &str = "lit_i64";
const TAG_LIT_F64: &str = "lit_f64";
const TAG_LIT_NULL: &str = "lit_null";
const TAG_IS_IN: &str = "is_in";

const OP_EQ: &str = "eq";
const OP_NEQ: &str = "neq";
const OP_LT: &str = "lt";
const OP_LTE: &str = "lte";
const OP_GT: &str = "gt";
const OP_GTE: &str = "gte";
const OP_AND: &str = "and";
const OP_OR: &str = "or";

fn write_json_escaped(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
}

fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    write_json_escaped(out, s);
    out.push('"');
}

fn anyvalue_to_json(av: AnyValue, dtype: &DataType, out: &mut String) -> Option<()> {
    match av {
        AnyValue::String(s) => {
            write!(out, "[\"{TAG_LIT_STR}\",").unwrap();
            write_json_string(out, s);
            out.push(']');
        },
        AnyValue::Boolean(val) => {
            if val {
                write!(out, "[\"{TAG_LIT_BOOL}\",true]").unwrap();
            } else {
                write!(out, "[\"{TAG_LIT_BOOL}\",false]").unwrap();
            }
        },
        AnyValue::Null => {
            write!(out, "[\"{TAG_LIT_NULL}\"]").unwrap();
        },
        // TODO: This should get improved to support more numerics
        av => {
            if dtype.is_float() {
                let val = av.extract::<f64>()?;
                write!(out, "[\"{TAG_LIT_F64}\",{val}]").unwrap();
            } else if dtype.is_integer() {
                let val = av.extract::<i64>()?;
                write!(out, "[\"{TAG_LIT_I64}\",{val}]").unwrap();
            } else {
                return None;
            }
        },
    }
    Some(())
}

fn op_to_json_str(op: &Operator) -> Option<&'static str> {
    match op {
        Operator::Eq | Operator::EqValidity => Some(OP_EQ),
        Operator::NotEq | Operator::NotEqValidity => Some(OP_NEQ),
        Operator::Lt => Some(OP_LT),
        Operator::LtEq => Some(OP_LTE),
        Operator::Gt => Some(OP_GT),
        Operator::GtEq => Some(OP_GTE),
        Operator::And => Some(OP_AND),
        Operator::Or => Some(OP_OR),
        _ => None,
    }
}

pub fn predicate_to_pa(predicate: Node, expr_arena: &Arena<AExpr>) -> Option<String> {
    let mut out = String::new();
    predicate_to_pa_json(predicate, expr_arena, &mut out)?;
    Some(out)
}

fn predicate_to_pa_json(
    predicate: Node,
    expr_arena: &Arena<AExpr>,
    out: &mut String,
) -> Option<()> {
    match expr_arena.get(predicate) {
        AExpr::BinaryExpr { left, right, op } => {
            // Only built to support comparison and bool operators. I can't really see
            // a reason to support math ops at this time.
            if !op.is_comparison_or_bitwise() {
                return None;
            }
            let op_str: &str = op_to_json_str(op)?;
            write!(out, "[\"{TAG_BINOP}\",\"{op_str}\",").unwrap();

            predicate_to_pa_json(*left, expr_arena, out)?;
            out.push(',');
            predicate_to_pa_json(*right, expr_arena, out)?;
            out.push(']');

            Some(())
        },

        AExpr::Column(name) => {
            write!(out, "[\"{TAG_FIELD}\",").unwrap();
            write_json_string(out, name);
            out.push(']');
            Some(())
        },

        AExpr::Literal(lv) => {
            let av = lv.to_any_value()?;
            let dtype: DataType = av.dtype();
            anyvalue_to_json(av, &dtype, out)
        },

        // This is for expressions that evaluate to bools making them predicates.
        // The most critical are probably null checks. In the future these should be expanded
        // to support the full range of bool expressions and probably a more general call.
        AExpr::Function {
            function, input, ..
        } => {
            let input_node = input.first().unwrap().node();

            match function {
                IRFunctionExpr::Boolean(IRBooleanFunction::IsNull) => {
                    write!(out, "[\"{TAG_IS_NULL}\",").unwrap();
                    predicate_to_pa_json(input_node, expr_arena, out)?;
                    out.push(']');
                    Some(())
                },
                IRFunctionExpr::Boolean(IRBooleanFunction::IsNotNull) => {
                    write!(out, "[\"{TAG_IS_NOT_NULL}\",").unwrap();
                    predicate_to_pa_json(input_node, expr_arena, out)?;
                    out.push(']');
                    Some(())
                },
                #[cfg(feature = "is_in")]
                IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { .. }) => {
                    let rhs_node = input.get(1)?.node();

                    // The RHS of is_in is a literal list/series. We need to
                    // extract individual elements and serialize them.
                    let AExpr::Literal(lv) = expr_arena.get(rhs_node) else {
                        return None;
                    };

                    let av = lv.to_any_value()?;
                    let s = match av {
                        AnyValue::List(s) => s,
                        _ => return None,
                    };

                    // Don't push down very large lists
                    if s.len() > 100 {
                        return None;
                    }

                    let dtype = s.dtype().clone();

                    write!(out, "[\"{TAG_IS_IN}\",").unwrap();
                    predicate_to_pa_json(input_node, expr_arena, out)?;
                    out.push_str(",[");
                    for (i, val) in s.iter().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        anyvalue_to_json(val, &dtype, out)?;
                    }
                    out.push_str("]]");
                    Some(())
                },
                _ => None,
            }
        },

        _ => None,
    }
}
