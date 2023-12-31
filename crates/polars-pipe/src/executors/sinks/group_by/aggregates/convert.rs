use std::any::Any;
use std::sync::Arc;

use polars_core::datatypes::Field;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{DataType, SchemaRef, Series, IDX_DTYPE};
use polars_core::schema::Schema;
use polars_io::predicates::PhysicalIoExpr;
use polars_plan::dsl::Expr;
use polars_plan::logical_plan::{ArenaExprIter, Context};
use polars_plan::prelude::{AAggExpr, AExpr};
use polars_utils::arena::{Arena, Node};
use polars_utils::IdxSize;

use crate::executors::sinks::group_by::aggregates::count::CountAgg;
use crate::executors::sinks::group_by::aggregates::first::FirstAgg;
use crate::executors::sinks::group_by::aggregates::last::LastAgg;
#[cfg(feature = "dtype-date")]
use crate::executors::sinks::group_by::aggregates::mean::Conversion;
use crate::executors::sinks::group_by::aggregates::mean::MeanAgg;
use crate::executors::sinks::group_by::aggregates::min_max::{new_max, new_min};
use crate::executors::sinks::group_by::aggregates::null::NullAgg;
use crate::executors::sinks::group_by::aggregates::{AggregateFunction, SumAgg};
use crate::expressions::PhysicalPipedExpr;
use crate::operators::DataChunk;

struct Count {}

impl PhysicalIoExpr for Count {
    fn evaluate_io(&self, _df: &DataFrame) -> PolarsResult<Series> {
        unimplemented!()
    }
}
impl PhysicalPipedExpr for Count {
    fn evaluate(&self, chunk: &DataChunk, _lazy_state: &dyn Any) -> PolarsResult<Series> {
        // the length must match the chunks as the operators expect that
        // so we fill a null series.
        Ok(Series::new_null("", chunk.data.height()))
    }

    fn field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        todo!()
    }

    fn expression(&self) -> Expr {
        Expr::Count
    }
}

pub fn can_convert_to_hash_agg(
    mut node: Node,
    expr_arena: &Arena<AExpr>,
    input_schema: &Schema,
) -> bool {
    let mut can_run_partitioned = true;
    if expr_arena
        .iter(node)
        .map(|(_, ae)| {
            match ae {
                AExpr::Agg(_)
                | AExpr::Count
                | AExpr::Cast { .. }
                | AExpr::Literal(_)
                | AExpr::Column(_)
                | AExpr::BinaryExpr { .. }
                | AExpr::Ternary { .. }
                | AExpr::Alias(_, _) => {},
                _ => {
                    can_run_partitioned = false;
                },
            }
            ae
        })
        .filter(|ae| matches!(ae, AExpr::Agg(_) | AExpr::Count))
        .count()
        == 1
        && can_run_partitioned
    {
        // last expression must be agg or agg.alias
        if let AExpr::Alias(input, _) = expr_arena.get(node) {
            node = *input
        }
        match expr_arena.get(node) {
            AExpr::Count => true,
            ae @ AExpr::Agg(agg_fn) => {
                matches!(
                    agg_fn,
                    AAggExpr::Sum(_)
                        | AAggExpr::First(_)
                        | AAggExpr::Last(_)
                        | AAggExpr::Mean(_)
                        | AAggExpr::Count(_, false)
                ) || (matches!(
                    agg_fn,
                    AAggExpr::Max {
                        propagate_nans: false,
                        ..
                    } | AAggExpr::Min {
                        propagate_nans: false,
                        ..
                    }
                ) && {
                    if let Ok(field) = ae.to_field(input_schema, Context::Default, expr_arena) {
                        field.dtype.to_physical().is_numeric()
                    } else {
                        false
                    }
                })
            },
            _ => false,
        }
    } else {
        false
    }
}

/// # Returns:
///  - input_dtype: dtype that goes into the agg expression
///  - physical expr: physical expression that produces the input of the aggregation
///  - aggregation function: the aggregation function
pub(crate) fn convert_to_hash_agg<F>(
    node: Node,
    expr_arena: &Arena<AExpr>,
    schema: &SchemaRef,
    to_physical: &F,
) -> (DataType, Arc<dyn PhysicalPipedExpr>, AggregateFunction)
where
    F: Fn(Node, &Arena<AExpr>, Option<&SchemaRef>) -> PolarsResult<Arc<dyn PhysicalPipedExpr>>,
{
    match expr_arena.get(node) {
        AExpr::Alias(input, _) => convert_to_hash_agg(*input, expr_arena, schema, to_physical),
        AExpr::Count => (
            IDX_DTYPE,
            Arc::new(Count {}),
            AggregateFunction::Count(CountAgg::new()),
        ),
        AExpr::Agg(agg) => match agg {
            AAggExpr::Min { input, .. } => {
                let phys_expr = to_physical(*input, expr_arena, Some(schema)).unwrap();
                let logical_dtype = phys_expr.field(schema).unwrap().dtype;

                let agg_fn = match logical_dtype.to_physical() {
                    DataType::Int8 => AggregateFunction::MinMaxI8(new_min()),
                    DataType::Int16 => AggregateFunction::MinMaxI16(new_min()),
                    DataType::Int32 => AggregateFunction::MinMaxI32(new_min()),
                    DataType::Int64 => AggregateFunction::MinMaxI64(new_min()),
                    DataType::UInt8 => AggregateFunction::MinMaxU8(new_min()),
                    DataType::UInt16 => AggregateFunction::MinMaxU16(new_min()),
                    DataType::UInt32 => AggregateFunction::MinMaxU32(new_min()),
                    DataType::UInt64 => AggregateFunction::MinMaxU64(new_min()),
                    DataType::Float32 => AggregateFunction::MinMaxF32(new_min()),
                    DataType::Float64 => AggregateFunction::MinMaxF64(new_min()),
                    dt => panic!("{dt} unexpected"),
                };
                (logical_dtype, phys_expr, agg_fn)
            },
            AAggExpr::Max { input, .. } => {
                let phys_expr = to_physical(*input, expr_arena, Some(schema)).unwrap();
                let logical_dtype = phys_expr.field(schema).unwrap().dtype;

                let agg_fn = match logical_dtype.to_physical() {
                    DataType::Int8 => AggregateFunction::MinMaxI8(new_max()),
                    DataType::Int16 => AggregateFunction::MinMaxI16(new_max()),
                    DataType::Int32 => AggregateFunction::MinMaxI32(new_max()),
                    DataType::Int64 => AggregateFunction::MinMaxI64(new_max()),
                    DataType::UInt8 => AggregateFunction::MinMaxU8(new_max()),
                    DataType::UInt16 => AggregateFunction::MinMaxU16(new_max()),
                    DataType::UInt32 => AggregateFunction::MinMaxU32(new_max()),
                    DataType::UInt64 => AggregateFunction::MinMaxU64(new_max()),
                    DataType::Float32 => AggregateFunction::MinMaxF32(new_max()),
                    DataType::Float64 => AggregateFunction::MinMaxF64(new_max()),
                    dt => panic!("{dt} unexpected"),
                };
                (logical_dtype, phys_expr, agg_fn)
            },
            AAggExpr::Sum(input) => {
                let phys_expr = to_physical(*input, expr_arena, Some(schema)).unwrap();
                let logical_dtype = phys_expr.field(schema).unwrap().dtype;

                #[cfg(feature = "dtype-categorical")]
                if matches!(logical_dtype, DataType::Categorical(_, _)) {
                    return (
                        logical_dtype.clone(),
                        phys_expr,
                        AggregateFunction::Null(NullAgg::new(logical_dtype)),
                    );
                }

                let agg_fn = match logical_dtype.to_physical() {
                    // Boolean is aggregated as the IDX type.
                    DataType::Boolean => {
                        if std::mem::size_of::<IdxSize>() == 4 {
                            AggregateFunction::SumU32(SumAgg::<u32>::new())
                        } else {
                            AggregateFunction::SumU64(SumAgg::<u64>::new())
                        }
                    },
                    // these are aggregated as i64 to prevent overflow
                    DataType::Int8 => AggregateFunction::SumI64(SumAgg::<i64>::new()),
                    DataType::Int16 => AggregateFunction::SumI64(SumAgg::<i64>::new()),
                    DataType::UInt8 => AggregateFunction::SumI64(SumAgg::<i64>::new()),
                    DataType::UInt16 => AggregateFunction::SumI64(SumAgg::<i64>::new()),
                    //  these stay true to there types
                    DataType::UInt32 => AggregateFunction::SumU32(SumAgg::<u32>::new()),
                    DataType::UInt64 => AggregateFunction::SumU64(SumAgg::<u64>::new()),
                    DataType::Int32 => AggregateFunction::SumI32(SumAgg::<i32>::new()),
                    DataType::Int64 => AggregateFunction::SumI64(SumAgg::<i64>::new()),
                    DataType::Float32 => AggregateFunction::SumF32(SumAgg::<f32>::new()),
                    DataType::Float64 => AggregateFunction::SumF64(SumAgg::<f64>::new()),
                    dt => AggregateFunction::Null(NullAgg::new(dt)),
                };
                (logical_dtype, phys_expr, agg_fn)
            },
            AAggExpr::Mean(input) => {
                let phys_expr = to_physical(*input, expr_arena, Some(schema)).unwrap();

                let logical_dtype = phys_expr.field(schema).unwrap().dtype;
                match &logical_dtype {
                    #[cfg(feature = "dtype-categorical")]
                    &DataType::Categorical(_, _) => (
                        logical_dtype.clone(),
                        phys_expr,
                        AggregateFunction::Null(NullAgg::new(logical_dtype)),
                    ),
                    &DataType::Date => (
                        logical_dtype,
                        to_physical(*input, expr_arena, Some(schema)).unwrap(),
                        AggregateFunction::MeanDate(MeanAgg::<i64>::new_date()),
                    ),
                    dt => {
                        let agg_fn = match dt.to_physical() {
                            dt if dt.is_integer() => {
                                AggregateFunction::MeanF64(MeanAgg::<f64>::new())
                            },
                            DataType::Float32 => AggregateFunction::MeanF32(MeanAgg::<f32>::new()),
                            DataType::Float64 => AggregateFunction::MeanF64(MeanAgg::<f64>::new()),
                            dt => AggregateFunction::Null(NullAgg::new(dt)),
                        };
                        (logical_dtype, phys_expr, agg_fn)
                    },
                }
            },
            AAggExpr::First(input) => {
                let phys_expr = to_physical(*input, expr_arena, Some(schema)).unwrap();
                let logical_dtype = phys_expr.field(schema).unwrap().dtype;
                (
                    logical_dtype.clone(),
                    phys_expr,
                    AggregateFunction::First(FirstAgg::new(logical_dtype.to_physical())),
                )
            },
            AAggExpr::Last(input) => {
                let phys_expr = to_physical(*input, expr_arena, Some(schema)).unwrap();
                let logical_dtype = phys_expr.field(schema).unwrap().dtype;
                (
                    logical_dtype.clone(),
                    phys_expr,
                    AggregateFunction::Last(LastAgg::new(logical_dtype.to_physical())),
                )
            },
            AAggExpr::Count(input, _) => {
                let phys_expr = to_physical(*input, expr_arena, Some(schema)).unwrap();
                let logical_dtype = phys_expr.field(schema).unwrap().dtype;
                (
                    logical_dtype,
                    phys_expr,
                    AggregateFunction::Count(CountAgg::new()),
                )
            },
            agg => panic!("{agg:?} not yet implemented."),
        },
        _ => todo!(),
    }
}
