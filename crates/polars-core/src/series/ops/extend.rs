use crate::prelude::*;

impl Series {
    /// Extend with a constant value.
    pub fn extend_constant(&self, value: AnyValue, n: usize) -> PolarsResult<Self> {
        let s =
            Series::from_any_values_and_dtype(PlSmallStr::EMPTY, &[value], self.dtype(), false)?;
        let to_append = s.new_from_index(0, n);

        let mut out = self.clone();
        out.append(&to_append)?;
        Ok(out)
    }
}
