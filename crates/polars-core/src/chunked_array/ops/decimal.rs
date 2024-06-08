use crate::chunked_array::cast::CastOptions;
use crate::prelude::*;

impl StringChunked {
    /// Convert an [`StringChunked`] to a [`Series`] of [`DataType::Decimal`].
    /// Scale needed for the decimal type are inferred.  Parsing is not strict.  
    /// Scale inference assumes that all tested strings are well-formed numbers,
    /// and may produce unexpected results for scale if this is not the case.
    ///
    /// If the decimal `precision` and `scale` are already known, consider
    /// using the `cast` method.
    pub fn to_decimal(&self, infer_length: usize) -> PolarsResult<Series> {
        let mut scale = 0;
        let mut iter = self.into_iter();
        let mut valid_count = 0;
        while let Some(Some(v)) = iter.next() {
            let scale_value = arrow::compute::decimal::infer_scale(v.as_bytes());
            scale = std::cmp::max(scale, scale_value);
            valid_count += 1;
            if valid_count == infer_length {
                break;
            }
        }

        self.cast_with_options(
            &DataType::Decimal(None, Some(scale as usize)),
            CastOptions::NonStrict,
        )
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_inferred_length() {
        use super::*;
        let vals = [
            "1.0",
            "invalid",
            "225.0",
            "3.00045",
            "-4.0",
            "5.104",
            "5.25251525353",
        ];
        let s = StringChunked::from_slice("test", &vals);
        let s = s.to_decimal(6).unwrap();
        assert_eq!(s.dtype(), &DataType::Decimal(None, Some(5)));
        assert_eq!(s.len(), 7);
        assert_eq!(s.get(0).unwrap(), AnyValue::Decimal(100000, 5));
        assert_eq!(s.get(1).unwrap(), AnyValue::Null);
        assert_eq!(s.get(3).unwrap(), AnyValue::Decimal(300045, 5));
        assert_eq!(s.get(4).unwrap(), AnyValue::Decimal(-400000, 5));
        assert_eq!(s.get(6).unwrap(), AnyValue::Decimal(525251, 5));
    }
}
