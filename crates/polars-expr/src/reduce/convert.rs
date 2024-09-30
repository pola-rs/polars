use polars_core::error::feature_gated;
use polars_plan::prelude::*;
use polars_utils::arena::{Arena, Node};

use super::len::LenReduce;
use super::mean::MeanReduce;
use super::min_max::{MaxReduce, MinReduce};
#[cfg(feature = "propagate_nans")]
use super::nan_min_max::{NanMaxReduce, NanMinReduce};
use super::sum::SumReduce;
use super::*;

/// Converts a node into a reduction + its associated selector expression.
pub fn into_reduction(
    node: Node,
    expr_arena: &mut Arena<AExpr>,
    schema: &Schema,
) -> PolarsResult<(Box<dyn Reduction>, Node)> {
    let get_dt = |node| {
        expr_arena
            .get(node)
            .to_dtype(schema, Context::Default, expr_arena)
    };
    let out = match expr_arena.get(node) {
        AExpr::Agg(agg) => match agg {
            IRAggExpr::Sum(input) => (
                Box::new(SumReduce::new(get_dt(*input)?)) as Box<dyn Reduction>,
                *input,
            ),
            IRAggExpr::Min {
                propagate_nans,
                input,
            } => {
                let dt = get_dt(*input)?;
                if *propagate_nans && dt.is_float() {
                    feature_gated!("propagate_nans", {
                        let out: Box<dyn Reduction> = match dt {
                            DataType::Float32 => Box::new(NanMinReduce::<Float32Type>::new()),
                            DataType::Float64 => Box::new(NanMinReduce::<Float64Type>::new()),
                            _ => unreachable!(),
                        };
                        (out, *input)
                    })
                } else {
                    (
                        Box::new(MinReduce::new(dt.clone())) as Box<dyn Reduction>,
                        *input,
                    )
                }
            },
            IRAggExpr::Max {
                propagate_nans,
                input,
            } => {
                let dt = get_dt(*input)?;
                if *propagate_nans && dt.is_float() {
                    feature_gated!("propagate_nans", {
                        let out: Box<dyn Reduction> = match dt {
                            DataType::Float32 => Box::new(NanMaxReduce::<Float32Type>::new()),
                            DataType::Float64 => Box::new(NanMaxReduce::<Float64Type>::new()),
                            _ => unreachable!(),
                        };
                        (out, *input)
                    })
                } else {
                    (Box::new(MaxReduce::new(dt.clone())) as _, *input)
                }
            },
            IRAggExpr::Mean(input) => {
                let out: Box<dyn Reduction> = Box::new(MeanReduce::new(get_dt(*input)?));
                (out, *input)
            },
            _ => unreachable!(),
        },
        AExpr::Len => {
            // Compute length on the first column, or if none exist we'll use
            // a zero-length dummy series.
            let out: Box<dyn Reduction> = Box::new(LenReduce::new());
            let expr = if let Some(first_column) = schema.iter_names().next() {
                expr_arena.add(AExpr::Column(first_column.as_str().into()))
            } else {
                let dummy = Series::new_null(PlSmallStr::from_static("dummy"), 0);
                expr_arena.add(AExpr::Literal(LiteralValue::Series(SpecialEq::new(dummy))))
            };
            (out, expr)
        },
        _ => unreachable!(),
    };
    Ok(out)
}
