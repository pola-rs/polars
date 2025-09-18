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
    pub fn to_decimal_infer(&self, infer_length: usize) -> PolarsResult<Series> {
        let mut scale = 0;
        let mut prec = 0;
        let mut iter = self.into_iter();
        let mut valid_count = 0;
        while let Some(Some(v)) = iter.next() {
            let mut bytes = v.as_bytes();
            if bytes.first() == Some(&b'-') {
                bytes = &bytes[1..];
            }
            if let Some(separator) = bytes.iter().position(|b| *b == b'.') {
                scale = scale.max(bytes.len() - 1 - separator);
                prec = prec.max(bytes.len() - 1);
            } else {
                prec = prec.max(bytes.len());
            }

            valid_count += 1;
            if valid_count == infer_length {
                break;
            }
        }

        self.to_decimal(prec, scale)
    }

    pub fn to_decimal(&self, prec: usize, scale: usize) -> PolarsResult<Series> {
        self.cast_with_options(&DataType::NewDecimal(prec, scale), CastOptions::NonStrict)
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
        let s = StringChunked::from_slice(PlSmallStr::from_str("test"), &vals);
        let s = s.to_decimal_infer(6).unwrap();
        assert_eq!(s.dtype(), &DataType::NewDecimal(12, 5));
        assert_eq!(s.len(), 7);
        assert_eq!(s.get(0).unwrap(), AnyValue::NewDecimal(100000, 12, 5));
        assert_eq!(s.get(1).unwrap(), AnyValue::Null);
        assert_eq!(s.get(3).unwrap(), AnyValue::NewDecimal(300045, 12, 5));
        assert_eq!(s.get(4).unwrap(), AnyValue::NewDecimal(-400000, 12, 5));
        assert_eq!(s.get(6).unwrap(), AnyValue::NewDecimal(525251, 12, 5));
    }
}
