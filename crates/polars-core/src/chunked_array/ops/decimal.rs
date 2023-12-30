use crate::prelude::*;

impl StringChunked {
    /// Convert an [`StringChunked`] to a [`Series`] of [`DataType::Decimal`].
    /// The parameters needed for the decimal type are inferred.
    ///
    /// If the decimal `precision` and `scale` are already known, consider
    /// using the `cast` method.
    pub fn to_decimal(&self, infer_length: usize) -> PolarsResult<Series> {
        let mut scale = 0;
        let mut iter = self.into_iter();
        let mut valid_count = 0;
        while let Some(Some(v)) = iter.next() {
            if let Some(scale_value) = arrow::legacy::compute::decimal::infer_scale(v.as_bytes()) {
                scale = std::cmp::max(scale, scale_value);
                valid_count += 1;
                if valid_count == infer_length {
                    break;
                }
            }
        }
        self.cast(&DataType::Decimal(None, Some(scale as usize)))
    }
}
