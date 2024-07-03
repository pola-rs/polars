use std::any::Any;
use arrow::legacy::error::PolarsResult;
use polars_core::datatypes::Scalar;
use polars_core::prelude::{AnyValue, DataType, Series};
use crate::reduce::Reduction;

pub(super) struct FirstReduce {
    value: Option<AnyValue<'static>>,
    dtype: DataType
}

impl FirstReduce {
    pub(super) fn new(dtype: DataType) -> Self {
        Self {
            value: None,
            dtype
        }
    }
}

impl Reduction for FirstReduce {
    fn init(&mut self) {
        self.value = None;
    }

    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        if matches!(self.value, AnyValue::Null) {

        }
        todo!()
    }

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()> {
        todo!()
    }

    fn finalize(&mut self) -> PolarsResult<Scalar> {
        todo!()
    }

    fn as_any(&self) -> &dyn Any {
        todo!()
    }
}