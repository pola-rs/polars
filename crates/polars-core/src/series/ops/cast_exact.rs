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
        // An integer converts to an integer or decimal exactly or not at all, and the non-strict
        // cast makes the values out of range null. The null counts tell whether any were.
        if self.dtype().is_integer() && (dtype.is_integer() || dtype.is_decimal()) {
            if casted.null_count() == self.null_count() {
                return Ok((casted, None));
            }
            let inexact = &casted.is_null() & &self.is_not_null();
            return Ok((casted, Some(inexact)));
        }
        // Null-safe: a cast back that overflows to null must count as inexact.
        let roundtrip = casted.cast_with_options(self.dtype(), CastOptions::NonStrict)?;
        let inexact = roundtrip.not_equal_missing(self)?;
        debug_assert_eq!(inexact.null_count(), 0);
        Ok((casted, inexact.any().then_some(inexact)))
    }
}
