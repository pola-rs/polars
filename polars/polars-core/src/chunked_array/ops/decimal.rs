use crate::prelude::*;

impl Utf8Chunked {
    /// Convert an [`Utf8Chunked`] to a `Series` of [`DataType::Decimal`].
    /// The parameters needed for the decimal type are inferred.
    ///
    /// If the decimal `precision` and `scale` are already known, consider
    /// using the `cast` method.
    pub fn to_decimal(&self, infer_length: usize) -> PolarsResult<Series> {
        let mut precision = 0;
        let mut scale = 0;
        let mut iter = self.into_iter();
        let mut valid_count = 0;
        while let Some(Some(v)) = iter.next() {
            if let Some(p) = polars_arrow::compute::decimal::infer_params(v.as_bytes()) {
                precision = std::cmp::max(precision, p.0);
                scale = std::cmp::max(scale, p.1);
                valid_count += 1;
                if valid_count == infer_length {
                    break;
                }
            }
        }
        polars_ensure!(precision > 0, ComputeError: "could not infer decimal parameters");
        self.cast(&DataType::Decimal(
            Some(precision as usize),
            Some(scale as usize),
        ))
    }
}
