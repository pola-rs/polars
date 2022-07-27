use polars_core::prelude::*;
use std::fmt::{Debug, Formatter};

pub trait DataFrameUdf: Send + Sync {
    fn call_udf(&self, df: DataFrame) -> Result<DataFrame>;
}

impl<F> DataFrameUdf for F
where
    F: Fn(DataFrame) -> Result<DataFrame> + Send + Sync,
{
    fn call_udf(&self, df: DataFrame) -> Result<DataFrame> {
        self(df)
    }
}

impl Debug for dyn DataFrameUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "dyn DataFrameUdf")
    }
}

pub trait UdfSchema: Send + Sync {
    fn get_schema(&self, input_schema: &Schema) -> Result<SchemaRef>;
}

impl<F> UdfSchema for F
where
    F: Fn(&Schema) -> Result<SchemaRef> + Send + Sync,
{
    fn get_schema(&self, input_schema: &Schema) -> Result<SchemaRef> {
        self(input_schema)
    }
}

impl Debug for dyn UdfSchema {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "dyn UdfSchema")
    }
}
