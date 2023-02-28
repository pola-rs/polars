use super::*;
use crate::prelude::*;

pub type DecimalChunked = Logical<DecimalType, Int128Type>;

impl Int128Chunked {
    #[inline]
    pub fn into_decimal_unchecked(self, precision: Option<usize>, scale: usize) -> DecimalChunked {
        let mut dt = DecimalChunked::new_logical(self);
        dt.2 = Some(DataType::Decimal(precision, Some(scale)));
        dt
    }

    pub fn into_decimal(
        self,
        precision: Option<usize>,
        scale: usize,
    ) -> PolarsResult<DecimalChunked> {
        // TODO: if prec is None, do we check that the value fits within precision of 38?...
        if let Some(precision) = precision {
            let prec_max = 10_i128.pow(precision as u32);
            // note: this is not too efficient as it scans through the data twice...
            if let (Some(min), Some(max)) = (self.min(), self.max()) {
                let max = max.abs().max(min.abs());
                if max > prec_max {
                    return Err(PolarsError::InvalidOperation(
                        format!(
                            "Decimal precision {precision} can't fit values with {} digits",
                            max.to_string().len(),
                        )
                        .into(),
                    ));
                }
            }
        }
        Ok(self.into_decimal_unchecked(precision, scale))
    }
}

impl LogicalType for DecimalChunked {
    fn dtype(&self) -> &DataType {
        self.2.as_ref().unwrap()
    }

    #[inline]
    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.0
            .get_any_value(i)
            .map(|av| av.into_decimal(self.scale()))
    }

    #[inline]
    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.0.get_any_value_unchecked(i).into_decimal(self.scale())
    }

    fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        let (prec_src, scale_src) = (self.precision(), self.scale());
        if let &DataType::Decimal(prec_dst, scale_dst) = dtype {
            let scale_dst = scale_dst.ok_or_else(|| {
                PolarsError::InvalidOperation("Cannot cast to Decimal with unknown scale".into())
            })?;
            // for now, let's just allow same-scale conversions
            // where precision is either the same or bigger or gets converted to `None`
            // (these are the easy cases requiring no checks and arithmetics which we can add later)
            let is_widen = match (prec_src, prec_dst) {
                (Some(prec_src), Some(prec_dst)) => prec_dst >= prec_src,
                (_, None) => true,
                _ => false,
            };
            if scale_src == scale_dst && is_widen {
                return self.0.cast(dtype); // no conversion or checks needed
            }
        }
        Err(PolarsError::InvalidOperation(
            format!("Cannot cast {} to {}", self.2.as_ref().unwrap(), dtype).into(),
        ))
    }
}

impl DecimalChunked {
    pub fn precision(&self) -> Option<usize> {
        match self.2.as_ref().unwrap() {
            DataType::Decimal(precision, _) => *precision,
            _ => unreachable!(),
        }
    }

    pub fn scale(&self) -> usize {
        match self.2.as_ref().unwrap() {
            DataType::Decimal(_, scale) => scale.unwrap_or_else(|| unreachable!()),
            _ => unreachable!(),
        }
    }
}
