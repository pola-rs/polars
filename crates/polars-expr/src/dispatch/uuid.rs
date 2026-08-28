use std::sync::Arc;

use polars_core::prelude::{
    Column, DataType, IntoColumn, IntoSeries, NewChunkedArray, PolarsResult, TimeUnit, TimeZone,
    UInt128Chunked,
};
use polars_plan::dsl::{ColumnsUdf, SpecialEq};
use polars_plan::plans::IRUuidFunction;

pub fn function_expr_to_udf(func: IRUuidFunction) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    match func {
        IRUuidFunction::GenerateV4 => map!(generate_v4),
        IRUuidFunction::GenerateV7 => map!(generate_v7),
        IRUuidFunction::Version => map!(version),
        #[cfg(feature = "dtype-datetime")]
        IRUuidFunction::Timestamp { strict } => map!(timestamp, strict),
    }
}

fn generate_v4(column: &Column) -> PolarsResult<Column> {
    let values = (0..column.len()).map(|_| ::uuid::Uuid::new_v4().as_u128());
    Ok(
        UInt128Chunked::from_iter_values(column.name().clone(), values)
            .into_uuid()
            .into_column(),
    )
}

fn generate_v7(column: &Column) -> PolarsResult<Column> {
    let values = (0..column.len()).map(|_| ::uuid::Uuid::now_v7().as_u128());
    let mut out = UInt128Chunked::from_iter_values(column.name().clone(), values).into_uuid();
    out.phys
        .set_sorted_flag(polars_core::series::IsSorted::Ascending);
    Ok(out.into_column())
}

fn version(column: &Column) -> PolarsResult<Column> {
    Ok(column
        .as_materialized_series()
        .uuid()?
        .version()
        .into_column())
}

#[cfg(feature = "dtype-datetime")]
fn timestamp(column: &Column, strict: bool) -> PolarsResult<Column> {
    let out = column
        .as_materialized_series()
        .uuid()?
        .timestamp_ms(strict)?
        .into_series()
        .cast(&DataType::Datetime(
            TimeUnit::Milliseconds,
            Some(TimeZone::UTC),
        ))?;
    Ok(out.into_column())
}
