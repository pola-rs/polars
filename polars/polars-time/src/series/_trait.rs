use super::*;
#[cfg(feature = "rolling_window")]
use crate::prelude::*;

#[cfg(feature = "rolling_window")]
macro_rules! invalid_operation {
    ($s:expr) => {
        Err(PolarsError::InvalidOperation(
            format!(
                "this operation is not implemented/valid for this dtype: {:?}",
                $s.ops_time_dtype()
            )
            .into(),
        ))
    };
}

pub trait SeriesOpsTime {
    fn ops_time_dtype(&self) -> &DataType;

    /// Apply a rolling mean to a Series. See:
    /// [ChunkedArray::rolling_mean](crate::prelude::ChunkWindow::rolling_mean).
    #[cfg(feature = "rolling_window")]
    fn rolling_mean(&self, _options: RollingOptionsImpl) -> PolarsResult<Series> {
        invalid_operation!(self)
    }
    /// Apply a rolling sum to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_sum(&self, _options: RollingOptionsImpl) -> PolarsResult<Series> {
        invalid_operation!(self)
    }
    /// Apply a rolling median to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_median(&self, _options: RollingOptionsImpl) -> PolarsResult<Series> {
        invalid_operation!(self)
    }
    /// Apply a rolling quantile to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_quantile(
        &self,
        _quantile: f64,
        _interpolation: QuantileInterpolOptions,
        _options: RollingOptionsImpl,
    ) -> PolarsResult<Series> {
        invalid_operation!(self)
    }

    /// Apply a rolling min to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_min(&self, _options: RollingOptionsImpl) -> PolarsResult<Series> {
        invalid_operation!(self)
    }
    /// Apply a rolling max to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_max(&self, _options: RollingOptionsImpl) -> PolarsResult<Series> {
        invalid_operation!(self)
    }

    /// Apply a rolling variance to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_var(&self, _options: RollingOptionsImpl) -> PolarsResult<Series> {
        invalid_operation!(self)
    }

    /// Apply a rolling std_dev to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_std(&self, _options: RollingOptionsImpl) -> PolarsResult<Series> {
        invalid_operation!(self)
    }
}

impl SeriesOpsTime for Series {
    fn ops_time_dtype(&self) -> &DataType {
        self.deref().dtype()
    }
    #[cfg(feature = "rolling_window")]
    fn rolling_mean(&self, _options: RollingOptionsImpl) -> PolarsResult<Series> {
        self.to_ops().rolling_mean(_options)
    }
    #[cfg(feature = "rolling_window")]
    fn rolling_sum(&self, _options: RollingOptionsImpl) -> PolarsResult<Series> {
        self.to_ops().rolling_sum(_options)
    }
    #[cfg(feature = "rolling_window")]
    fn rolling_median(&self, _options: RollingOptionsImpl) -> PolarsResult<Series> {
        self.to_ops().rolling_median(_options)
    }
    /// Apply a rolling quantile to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_quantile(
        &self,
        quantile: f64,
        interpolation: QuantileInterpolOptions,
        options: RollingOptionsImpl,
    ) -> PolarsResult<Series> {
        self.to_ops()
            .rolling_quantile(quantile, interpolation, options)
    }

    #[cfg(feature = "rolling_window")]
    fn rolling_min(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        self.to_ops().rolling_min(options)
    }
    /// Apply a rolling max to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_max(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        self.to_ops().rolling_max(options)
    }

    /// Apply a rolling variance to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_var(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        self.to_ops().rolling_var(options)
    }

    /// Apply a rolling std_dev to a Series.
    #[cfg(feature = "rolling_window")]
    fn rolling_std(&self, options: RollingOptionsImpl) -> PolarsResult<Series> {
        self.to_ops().rolling_std(options)
    }
}
