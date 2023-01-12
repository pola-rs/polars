use crate::prelude::*;

impl Series {
    /// Extend with a constant value.
    pub fn extend_constant(&self, value: AnyValue, n: usize) -> PolarsResult<Self> {
        let s = Series::from_any_values("", &[value]).unwrap();
        let s = s.cast(self.dtype())?;
        let to_append = s.new_from_index(0, n);

        let mut out = self.clone();
        out.append(&to_append)?;
        Ok(out)
    }
}
