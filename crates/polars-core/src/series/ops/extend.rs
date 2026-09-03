use crate::chunked_array::cast::CastOptions;
use crate::prelude::*;

impl Series {
    /// Extend with a constant value.
    pub fn extend_constant(&self, value: AnyValue, n: usize) -> PolarsResult<Self> {
        let scalar = Scalar::new(value.dtype(), value.into_static());
        let scalar = scalar.cast_with_options(self.dtype(), CastOptions::NonStrict)?;
        let s = Series::from_any_values_and_dtype(
            PlSmallStr::EMPTY,
            &[scalar.value().clone()],
            self.dtype(),
            true,
        )?;
        let to_append = s.new_from_index(0, n);

        let mut out = self.clone();
        out.append(&to_append)?;
        Ok(out)
    }
}
