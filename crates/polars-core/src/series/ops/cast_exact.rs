use crate::chunked_array::cast::CastOptions;
use crate::prelude::*;

impl Series {
    /// Cast non-strictly, also returning the rows the cast did not represent exactly.
    ///
    /// A row is inexact if it became null or does not survive the cast back unchanged. The mask
    /// is `None` if every row cast exactly.
    pub fn cast_reporting_inexact(
        &self,
        dtype: &DataType,
    ) -> PolarsResult<(Series, Option<BooleanChunked>)> {
        let casted = self.cast_with_options(dtype, CastOptions::NonStrict)?;
        // Null-safe: a cast back that overflows to null must count as inexact.
        let roundtrip = casted.cast_with_options(self.dtype(), CastOptions::NonStrict)?;
        let inexact = roundtrip.not_equal_missing(self)?;
        debug_assert_eq!(inexact.null_count(), 0);
        Ok((casted, inexact.any().then_some(inexact)))
    }
}
