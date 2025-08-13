use polars_core::error::{PolarsResult, feature_gated, polars_bail};
use polars_core::prelude::{AnyValue, DataType, Field, NamedFrom};
use polars_core::scalar::Scalar;
use polars_core::series::Series;
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use super::ExprToIRContext;
use crate::constants::get_literal_name;
use crate::dsl::{DataTypeFunction, SpecialEq, StructDataTypeFunction};
use crate::plans::{AExpr, ExprIR, IRFunctionExpr, LiteralValue, OutputName};

pub fn datatype_fn_to_aexpr(
    f: DataTypeFunction,
    ctx: &mut ExprToIRContext,
) -> PolarsResult<(AExpr, PlSmallStr)> {
    use DataTypeFunction as DTF;
    Ok(match f {
        DTF::Display(dt) => (
            AExpr::Literal(LiteralValue::Scalar(Scalar::from(PlSmallStr::from_string(
                dt.into_datatype(ctx.schema)?.to_string(),
            )))),
            get_literal_name().clone(),
        ),
        DTF::Eq(l, r) => (
            AExpr::Literal(LiteralValue::Scalar(Scalar::from(
                l.into_datatype(ctx.schema)? == r.into_datatype(ctx.schema)?,
            ))),
            get_literal_name().clone(),
        ),
        DTF::Matches(dt, selector) => {
            let dt = dt.into_datatype(ctx.schema)?;
            (
                AExpr::Literal(LiteralValue::Scalar(Scalar::from(selector.matches(&dt)))),
                get_literal_name().clone(),
            )
        },
        DTF::DefaultValue {
            dt_expr,
            n,
            numeric_to_one,
            num_list_values,
        } => {
            let dt = dt_expr.into_datatype(ctx.schema)?;
            let value = AnyValue::default_value(&dt, numeric_to_one, num_list_values);

            let mut expr = AExpr::Literal(LiteralValue::Scalar(Scalar::new(dt, value)));

            if n != 1 {
                let function = IRFunctionExpr::Repeat;
                let options = function.function_options();
                expr = AExpr::Function {
                    input: vec![
                        ExprIR::new(ctx.arena.add(expr), OutputName::Alias(Default::default())),
                        ExprIR::new(
                            ctx.arena
                                .add(AExpr::Literal(LiteralValue::Scalar(Scalar::from(
                                    n as IdxSize,
                                )))),
                            OutputName::Alias(Default::default()),
                        ),
                    ],
                    function,
                    options,
                };
            }

            (expr, get_literal_name().clone())
        },
        DTF::Array(dt_expr, f) => {
            let (inner, width): (DataType, usize) = match dt_expr.into_datatype(ctx.schema)? {
                #[cfg(feature = "dtype-array")]
                DataType::Array(inner, width) => (*inner, width),
                dt => polars_bail!(InvalidOperation: "`{dt}` is not an Array"),
            };

            feature_gated!("dtype-array", {
                use crate::dsl::ArrayDataTypeFunction;
                let value = match f {
                    ArrayDataTypeFunction::Width => {
                        LiteralValue::Scalar(Scalar::from(width as u32))
                    },
                    ArrayDataTypeFunction::Shape => {
                        let mut dims = vec![width as u32];
                        let mut inner = inner;
                        while let DataType::Array(new_inner, width) = inner {
                            dims.push(width as u32);
                            inner = *new_inner;
                        }
                        LiteralValue::Series(SpecialEq::new(Series::new(
                            get_literal_name().clone(),
                            dims,
                        )))
                    },
                };
                let value = AExpr::Literal(value);
                (value, get_literal_name().clone())
            })
        },
        DTF::Struct(dt_expr, f) => {
            let fields: Vec<Field> = match dt_expr.into_datatype(ctx.schema)? {
                #[cfg(feature = "dtype-struct")]
                DataType::Struct(fields) => fields,
                dt => polars_bail!(InvalidOperation: "`{dt}` is not an Struct"),
            };

            let value = match f {
                StructDataTypeFunction::FieldNames => {
                    LiteralValue::Series(SpecialEq::new(Series::new(
                        get_literal_name().clone(),
                        fields
                            .iter()
                            .map(|f| f.name.as_str())
                            .collect::<Vec<&str>>(),
                    )))
                },
            };
            let value = AExpr::Literal(value);
            (value, get_literal_name().clone())
        },
    })
}
