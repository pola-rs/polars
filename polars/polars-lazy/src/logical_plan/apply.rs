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
        write!(f, "udf")
    }
}
