use std::borrow::Cow;

use super::*;
use crate::chunked_array::cast::cast_chunks;
use crate::prelude::*;

pub type DecimalChunked = Logical<DecimalType, Int128Type>;

impl Int128Chunked {
    #[inline]
    pub fn into_decimal_unchecked(self, precision: Option<usize>, scale: usize) -> DecimalChunked {
        let mut dt = DecimalChunked::new_logical(self);
        dt.dtype = Some(DataType::Decimal(precision, Some(scale)));
        dt
    }

    pub fn into_decimal(
        self,
        precision: Option<usize>,
        scale: usize,
    ) -> PolarsResult<DecimalChunked> {
        // TODO: if precision is None, do we check that the value fits within precision of 38?...
        if let Some(precision) = precision {
            let precision_max = 10_i128.pow(precision as u32);
            if let Some((min, max)) = self.min_max() {
                let max_abs = max.abs().max(min.abs());
                polars_ensure!(
                    max_abs < precision_max,
                    ComputeError: "decimal precision {} can't fit values with {} digits",
                    precision,
                    max_abs.to_string().len()
                );
            }
        }
        Ok(self.into_decimal_unchecked(precision, scale))
    }
}

impl LogicalType for DecimalChunked {
    fn dtype(&self) -> &DataType {
        self.dtype.as_ref().unwrap()
    }

    #[inline]
    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        polars_ensure!(i < self.len(), oob = i, self.len());
        Ok(unsafe { self.get_any_value_unchecked(i) })
    }

    #[inline]
    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        match self.phys.get_unchecked(i) {
            Some(v) => AnyValue::Decimal(v, self.scale()),
            None => AnyValue::Null,
        }
    }

    fn cast_with_options(
        &self,
        dtype: &DataType,
        cast_options: CastOptions,
    ) -> PolarsResult<Series> {
        let mut dtype = Cow::Borrowed(dtype);
        if let DataType::Decimal(to_precision, to_scale) = dtype.as_ref() {
            let from_precision = self.precision();
            let from_scale = self.scale();

            let to_precision = to_precision.or(from_precision);
            let to_scale = to_scale.unwrap_or(from_scale);

            if to_precision == from_precision && to_scale == from_scale {
                return Ok(self.clone().into_series());
            }

            dtype = Cow::Owned(DataType::Decimal(to_precision, Some(to_scale)));
        }

        let arrow_dtype = self.dtype().to_arrow(CompatLevel::newest());
        let chunks = self
            .chunks
            .iter()
            .map(|arr| {
                arr.as_any()
                    .downcast_ref::<PrimitiveArray<i128>>()
                    .unwrap()
                    .clone()
                    .to(arrow_dtype.clone())
                    .to_boxed()
            })
            .collect::<Vec<_>>();
        let chunks = cast_chunks(&chunks, dtype.as_ref(), cast_options)?;
        Series::try_from((self.name().clone(), chunks))
    }
}

impl DecimalChunked {
    pub fn precision(&self) -> Option<usize> {
        match self.dtype.as_ref().unwrap() {
            DataType::Decimal(precision, _) => *precision,
            _ => unreachable!(),
        }
    }

    pub fn scale(&self) -> usize {
        match self.dtype.as_ref().unwrap() {
            DataType::Decimal(_, scale) => scale.unwrap_or_else(|| unreachable!()),
            _ => unreachable!(),
        }
    }

    pub fn to_scale(&self, scale: usize) -> PolarsResult<Cow<'_, Self>> {
        if self.scale() == scale {
            return Ok(Cow::Borrowed(self));
        }

        let mut precision = self.precision();
        if let Some(ref mut precision) = precision {
            if self.scale() < scale {
                *precision += scale;
                *precision = (*precision).min(38);
            }
        }

        let s = self.cast_with_options(
            &DataType::Decimal(precision, Some(scale)),
            CastOptions::NonStrict,
        )?;
        Ok(Cow::Owned(s.decimal().unwrap().clone()))
    }
}
