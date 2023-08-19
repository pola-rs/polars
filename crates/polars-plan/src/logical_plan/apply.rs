use std::fmt::{Debug, Formatter};

use polars_core::prelude::*;

pub trait DataFrameUdf: Send + Sync {
    fn call_udf(&self, df: DataFrame) -> PolarsResult<DataFrame>;
}

impl<F> DataFrameUdf for F
where
    F: Fn(DataFrame) -> PolarsResult<DataFrame> + Send + Sync,
{
    fn call_udf(&self, df: DataFrame) -> PolarsResult<DataFrame> {
        self(df)
    }
}

pub trait DataFrameUdfMut: Send + Sync {
    fn call_udf(&mut self, df: DataFrame) -> PolarsResult<DataFrame>;
}

impl<F> DataFrameUdfMut for F
where
    F: FnMut(DataFrame) -> PolarsResult<DataFrame> + Send + Sync,
{
    fn call_udf(&mut self, df: DataFrame) -> PolarsResult<DataFrame> {
        self(df)
    }
}

impl Debug for dyn DataFrameUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "dyn DataFrameUdf")
    }
}
impl Debug for dyn DataFrameUdfMut {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "dyn DataFrameUdfMut")
    }
}

pub trait UdfSchema: Send + Sync {
    fn get_schema(&self, input_schema: &Schema) -> PolarsResult<SchemaRef>;
}

impl<F> UdfSchema for F
where
    F: Fn(&Schema) -> PolarsResult<SchemaRef> + Send + Sync,
{
    fn get_schema(&self, input_schema: &Schema) -> PolarsResult<SchemaRef> {
        self(input_schema)
    }
}

impl Debug for dyn UdfSchema {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "dyn UdfSchema")
    }
}
