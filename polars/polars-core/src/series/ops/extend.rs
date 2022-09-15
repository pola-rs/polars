use crate::prelude::*;

impl Series {
    /// Extend with a constant value.
    pub fn extend_constant(&self, value: AnyValue, n: usize) -> PolarsResult<Self> {
        use AnyValue::*;
        let s = match value {
            Float32(v) => Series::new("", vec![v]),
            Float64(v) => Series::new("", vec![v]),
            UInt32(v) => Series::new("", vec![v]),
            UInt64(v) => Series::new("", vec![v]),
            Int32(v) => Series::new("", vec![v]),
            Int64(v) => Series::new("", vec![v]),
            Utf8(v) => Series::new("", vec![v]),
            Boolean(v) => Series::new("", vec![v]),
            Null => BooleanChunked::full_null("", 1).into_series(),
            dt => panic!("{:?} not supported", dt),
        };
        let s = s.cast(self.dtype())?;
        let to_append = s.expand_at_index(0, n);

        let mut out = self.clone();
        out.append(&to_append)?;
        Ok(out)
    }
}
