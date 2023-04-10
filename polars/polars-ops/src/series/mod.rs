mod ops;
pub use ops::*;
use polars_core::prelude::*;

pub trait IntoSeriesOps {
    fn to_series(&self) -> &Series;
}

impl IntoSeriesOps for Series {
    fn to_series(&self) -> &Series {
        self
    }
}

impl<T: IntoSeriesOps> SeriesOps for T {}

pub trait SeriesOps: IntoSeriesOps {
    #[cfg(feature = "approx_unique")]
    fn approx_unique(&self, precision: u8) -> PolarsResult<Series> {
        let s = self.to_series();
        approx_unique(s, precision)
    }
}
