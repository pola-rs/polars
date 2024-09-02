pub mod reduce;

use polars_utils::pl_str::PlSmallStr;

use crate::datatypes::{AnyValue, DataType};
use crate::prelude::Series;

#[derive(Clone)]
pub struct Scalar {
    dtype: DataType,
    value: AnyValue<'static>,
}

impl Scalar {
    #[inline(always)]
    pub fn new(dtype: DataType, value: AnyValue<'static>) -> Self {
        Self { dtype, value }
    }

    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.value.is_null()
    }

    #[inline(always)]
    pub fn value(&self) -> &AnyValue<'static> {
        &self.value
    }

    pub fn as_any_value(&self) -> AnyValue {
        self.value
            .strict_cast(&self.dtype)
            .unwrap_or_else(|| self.value.clone())
    }

    pub fn into_series(self, name: PlSmallStr) -> Series {
        Series::from_any_values_and_dtype(name, &[self.as_any_value()], &self.dtype, true).unwrap()
    }

    #[inline(always)]
    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    #[inline(always)]
    pub fn update(&mut self, value: AnyValue<'static>) {
        self.value = value;
    }
}
