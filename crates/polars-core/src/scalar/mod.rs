mod from;
mod new;
pub mod reduce;

use std::hash::Hash;

use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::chunked_array::cast::CastOptions;
use crate::datatypes::{AnyValue, DataType};
use crate::prelude::{Column, Series};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Scalar {
    dtype: DataType,
    value: AnyValue<'static>,
}

impl Hash for Scalar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        self.value.hash_impl(state, true);
    }
}

impl Default for Scalar {
    fn default() -> Self {
        Self {
            dtype: DataType::Null,
            value: AnyValue::Null,
        }
    }
}

impl Scalar {
    #[inline(always)]
    pub const fn new(dtype: DataType, value: AnyValue<'static>) -> Self {
        Self { dtype, value }
    }

    pub const fn null(dtype: DataType) -> Self {
        Self::new(dtype, AnyValue::Null)
    }

    pub fn cast_with_options(self, dtype: &DataType, options: CastOptions) -> PolarsResult<Self> {
        if self.dtype() == dtype {
            return Ok(self);
        }

        // @Optimize: If we have fully fleshed out casting semantics, we could just specify the
        // cast on AnyValue.
        let s = self
            .into_series(PlSmallStr::from_static("scalar"))
            .cast_with_options(dtype, options)?;
        let value = s.get(0).unwrap();
        Ok(Self::new(s.dtype().clone(), value.into_static()))
    }

    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.value.is_null()
    }

    #[inline(always)]
    pub fn is_nan(&self) -> bool {
        self.value.is_nan()
    }

    #[inline(always)]
    pub fn into_value(self) -> AnyValue<'static> {
        self.value
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

    /// Turn a scalar into a column with `length=1`.
    pub fn into_column(self, name: PlSmallStr) -> Column {
        Column::new_scalar(name, self, 1)
    }

    #[inline(always)]
    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    #[inline(always)]
    pub fn update(&mut self, value: AnyValue<'static>) {
        self.value = value;
    }

    #[inline(always)]
    pub fn with_value(mut self, value: AnyValue<'static>) -> Self {
        self.update(value);
        self
    }
}
