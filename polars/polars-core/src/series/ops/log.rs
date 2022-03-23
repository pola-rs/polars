use crate::prelude::*;

fn log<T: PolarsNumericType>(ca: &ChunkedArray<T>, base: f64) -> Float64Chunked {
    ca.cast_and_apply_in_place(|v: f64| v.log(base))
}

impl Series {
    /// Compute the logarithm to a given base
    #[cfg_attr(docsrs, doc(cfg(feature = "log")))]
    pub fn log(&self, base: f64) -> Series {
        let s = self.to_physical_repr();
        let s = s.as_ref();

        use DataType::*;
        match s.dtype() {
            Int32 => log(s.i32().unwrap(), base).into_series(),
            Int64 => log(s.i64().unwrap(), base).into_series(),
            UInt32 => log(s.u32().unwrap(), base).into_series(),
            UInt64 => log(s.u64().unwrap(), base).into_series(),
            Float32 => s.f32().unwrap().apply(|v| v.log(base as f32)).into_series(),
            Float64 => s.f64().unwrap().apply(|v| v.log(base)).into_series(),
            _ => unimplemented!(),
        }
    }

    /// Compute the entropy as `-sum(pk * log(pk)`.
    /// where `pk` are discrete probabilities.
    #[cfg_attr(docsrs, doc(cfg(feature = "log")))]
    pub fn entropy(&self, base: f64) -> Option<f64> {
        let pk = self;
        let log_pk = pk.log(base);
        (pk * &log_pk).sum::<f64>().map(|v| -v)
    }
}
