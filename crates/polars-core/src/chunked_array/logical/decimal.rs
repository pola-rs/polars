use std::borrow::Cow;

use arrow::bitmap::Bitmap;
use polars_compute::decimal::{
    DEC128_MAX_PREC, dec128_fits, dec128_mul, dec128_rescale, dec128_verify_prec_scale,
    i128_to_dec128,
};

use super::*;
use crate::chunked_array::cast::cast_chunks_from;
use crate::prelude::arity::{unary_elementwise, unary_kernel};
use crate::prelude::*;

pub type DecimalChunked = Logical<DecimalType, Int128Type>;

impl Int128Chunked {
    #[inline]
    pub fn into_decimal_unchecked(self, precision: usize, scale: usize) -> DecimalChunked {
        // SAFETY: no invalid states (from a safety perspective).
        unsafe { DecimalChunked::new_logical(self, DataType::Decimal(precision, scale)) }
    }

    pub fn into_decimal(self, precision: usize, scale: usize) -> PolarsResult<DecimalChunked> {
        dec128_verify_prec_scale(precision, scale)?;
        if let Some((min, max)) = self.min_max() {
            let max_abs = max.unsigned_abs().max(min.unsigned_abs());
            polars_ensure!(
                max_abs < i128::MAX as u128 && dec128_fits(max_abs as i128, precision),
                ComputeError: "decimal precision {} can't fit values with {} digits",
                precision,
                max_abs.to_string().len()
            );
        }
        Ok(self.into_decimal_unchecked(precision, scale))
    }
}

impl LogicalType for DecimalChunked {
    fn dtype(&self) -> &DataType {
        &self.dtype
    }

    #[inline]
    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        polars_ensure!(i < self.len(), oob = i, self.len());
        Ok(unsafe { self.get_any_value_unchecked(i) })
    }

    #[inline]
    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        match self.phys.get_unchecked(i) {
            Some(v) => AnyValue::Decimal(v, self.precision(), self.scale()),
            None => AnyValue::Null,
        }
    }

    fn cast_with_options(
        &self,
        dtype: &DataType,
        cast_options: CastOptions,
    ) -> PolarsResult<Series> {
        if let DataType::Decimal(to_prec, to_scale) = dtype {
            return Ok(self
                .with_prec_scale(*to_prec, *to_scale, cast_options.is_strict())?
                .into_owned()
                .into_series());
        }

        match dtype {
            DataType::Decimal(to_prec, to_scale) => {
                return Ok(self
                    .with_prec_scale(*to_prec, *to_scale, cast_options.is_strict())?
                    .into_owned()
                    .into_series());
            },

            dt if dt.is_primitive_numeric()
                | matches!(dt, DataType::String | DataType::Boolean) =>
            {
                // The chunks hold `i128`s and nothing that says what they are worth, so the
                // precision and scale are what the cast is told to read them as.
                let chunks =
                    cast_chunks_from(&self.physical().chunks, self.dtype(), dtype, cast_options)?;
                // SAFETY: the chunks were just cast to `dtype`, which the arm above restricts to
                // the types that are their own physical type.
                debug_assert_eq!(&dtype.to_physical(), dtype);
                Ok(unsafe {
                    Series::from_chunks_and_dtype_unchecked(self.name().clone(), chunks, dtype)
                })
            },

            dt => {
                polars_bail!(
                    InvalidOperation:
                    "casting from {:?} to {:?} not supported",
                    self.dtype(), dt
                )
            },
        }
    }
}

impl DecimalChunked {
    pub fn precision(&self) -> usize {
        match &self.dtype {
            DataType::Decimal(precision, _) => *precision,
            _ => unreachable!(),
        }
    }

    pub fn scale(&self) -> usize {
        match &self.dtype {
            DataType::Decimal(_, scale) => *scale,
            _ => unreachable!(),
        }
    }

    pub fn with_prec_scale(
        &self,
        prec: usize,
        scale: usize,
        strict: bool,
    ) -> PolarsResult<Cow<'_, Self>> {
        if self.precision() == prec && self.scale() == scale {
            return Ok(Cow::Borrowed(self));
        }

        dec128_verify_prec_scale(prec, scale)?;
        let phys = if self.scale() == scale {
            if prec >= self.precision() {
                // Increasing precision is always allowed.
                self.phys.clone()
            } else if strict {
                if let Some((min, max)) = self.phys.min_max() {
                    let max_abs = max.unsigned_abs().max(min.unsigned_abs());
                    polars_ensure!(
                        max_abs < i128::MAX as u128 && dec128_fits(max_abs as i128, prec),
                        ComputeError: "decimal precision {} can't fit values with {} digits",
                        prec,
                        max_abs.to_string().len()
                    );
                }
                self.phys.clone()
            } else {
                unary_kernel(&self.phys, |arr| {
                    let new_valid: Bitmap = arr
                        .iter()
                        .map(|opt_x| {
                            if let Some(x) = opt_x {
                                dec128_fits(x, prec)
                            } else {
                                false
                            }
                        })
                        .collect();
                    arr.clone()
                        .with_validity_typed(Some(PlBitmap::from_bitmap(new_valid)))
                })
            }
        } else {
            let old_s = self.scale();
            unary_elementwise(&self.phys, |x| dec128_rescale(x?, old_s, prec, scale))
        };

        let ca = unsafe { DecimalChunked::new_logical(phys, DataType::Decimal(prec, scale)) };
        Ok(Cow::Owned(ca))
    }

    /// Converts self to a physical representation with the given precision and
    /// scale, returning the given sentinel value instead for values which don't
    /// fit in the given precision and scale. This can be useful for comparisons.
    pub fn into_phys_with_prec_scale_or_sentinel(
        &self,
        prec: usize,
        scale: usize,
        sentinel: i128,
    ) -> Int128Chunked {
        if self.precision() <= prec && self.scale() == scale {
            return self.phys.clone();
        }

        let old_s = self.scale();
        unary_elementwise(&self.phys, |x| {
            Some(dec128_rescale(x?, old_s, prec, scale).unwrap_or(sentinel))
        })
    }

    pub fn prod_reduce(&self) -> Scalar {
        let prec = DEC128_MAX_PREC;
        let scale = self.scale();

        let mut prod = i128_to_dec128(1, prec, scale).unwrap();
        for arr in self.phys.downcast_iter() {
            for v in arr.iter().flatten() {
                if let Some(p) = dec128_mul(prod, v, prec, scale) {
                    prod = p;
                } else {
                    return Scalar::null(DataType::Decimal(prec, scale));
                }
            }
        }

        Scalar::new(
            DataType::Decimal(prec, scale),
            AnyValue::Decimal(prod, prec, scale),
        )
    }
}
