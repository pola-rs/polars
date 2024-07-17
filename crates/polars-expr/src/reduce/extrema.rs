#[cfg(feature = "propagate_nans")]
use polars_core::datatypes::PolarsFloatType;
#[cfg(feature = "propagate_nans")]
use polars_ops::prelude::nan_propagating_aggregate;
#[cfg(feature = "propagate_nans")]
use polars_utils::min_max::MinMax;

use super::*;

#[derive(Clone)]
pub(super) struct MinReduce {
    dtype: DataType,
    value: Option<Scalar>,
}

impl MinReduce {
    pub(super) fn new(dtype: DataType) -> Self {
        Self { dtype, value: None }
    }

    fn update_impl(&mut self, other: &AnyValue<'static>) {
        if let Some(value) = &mut self.value {
            if other < value.value() {
                value.update(other.clone());
            }
        } else {
            self.value = Some(Scalar::new(self.dtype.clone(), other.clone()))
        }
    }
}

impl Reduction for MinReduce {
    fn init_dyn(&self) -> Box<dyn Reduction> {
        Box::new(Self::new(self.dtype.clone()))
    }

    fn reset(&mut self) {
        *self = Self::new(self.dtype.clone());
    }

    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        let sc = batch.min_reduce()?;
        self.update_impl(sc.value());
        Ok(())
    }

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        if let Some(value) = &other.value {
            self.update_impl(value.value());
        }
        Ok(())
    }

    fn finalize(&mut self) -> PolarsResult<Scalar> {
        if let Some(value) = self.value.take() {
            Ok(value)
        } else {
            Ok(Scalar::new(self.dtype.clone(), AnyValue::Null))
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
#[derive(Clone)]
pub(super) struct MaxReduce {
    dtype: DataType,
    value: Option<Scalar>,
}

impl MaxReduce {
    pub(super) fn new(dtype: DataType) -> Self {
        Self { dtype, value: None }
    }
    fn update_impl(&mut self, other: &AnyValue<'static>) {
        if let Some(value) = &mut self.value {
            if other > value.value() {
                value.update(other.clone());
            }
        } else {
            self.value = Some(Scalar::new(self.dtype.clone(), other.clone()))
        }
    }
}

impl Reduction for MaxReduce {
    fn init_dyn(&self) -> Box<dyn Reduction> {
        Box::new(Self::new(self.dtype.clone()))
    }
    fn reset(&mut self) {
        *self = Self::new(self.dtype.clone());
    }

    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        let sc = batch.max_reduce()?;
        self.update_impl(sc.value());
        Ok(())
    }

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();

        if let Some(value) = &other.value {
            self.update_impl(value.value());
        }
        Ok(())
    }

    fn finalize(&mut self) -> PolarsResult<Scalar> {
        if let Some(value) = self.value.take() {
            Ok(value)
        } else {
            Ok(Scalar::new(self.dtype.clone(), AnyValue::Null))
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(feature = "propagate_nans")]
#[derive(Clone)]
pub(super) struct MaxNanReduce<T: PolarsFloatType> {
    value: Option<T::Native>,
}

#[cfg(feature = "propagate_nans")]
impl<T: PolarsFloatType> MaxNanReduce<T>
where
    T::Native: MinMax,
{
    pub(super) fn new() -> Self {
        Self { value: None }
    }
    fn update_impl(&mut self, other: T::Native) {
        if let Some(value) = self.value {
            self.value = Some(MinMax::max_propagate_nan(value, other));
        } else {
            self.value = Some(other);
        }
    }
}

#[cfg(feature = "propagate_nans")]
impl<T: PolarsFloatType + Clone> Reduction for MaxNanReduce<T>
where
    T::Native: MinMax,
{
    fn init_dyn(&self) -> Box<dyn Reduction> {
        Box::new(Self::new())
    }
    fn reset(&mut self) {
        self.value = None;
    }

    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        if let Some(v) = nan_propagating_aggregate::ca_nan_agg(
            batch.unpack::<T>().unwrap(),
            MinMax::max_propagate_nan,
        ) {
            self.update_impl(v)
        }
        Ok(())
    }

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();

        if let Some(value) = &other.value {
            self.update_impl(*value);
        }
        Ok(())
    }

    fn finalize(&mut self) -> PolarsResult<Scalar> {
        let av = AnyValue::from(self.value);
        Ok(Scalar::new(T::get_dtype(), av))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
#[cfg(feature = "propagate_nans")]
#[derive(Clone)]
pub(super) struct MinNanReduce<T: PolarsFloatType> {
    value: Option<T::Native>,
}

#[cfg(feature = "propagate_nans")]
impl<T: PolarsFloatType> crate::reduce::extrema::MinNanReduce<T>
where
    T::Native: MinMax,
{
    pub(super) fn new() -> Self {
        Self { value: None }
    }
    fn update_impl(&mut self, other: T::Native) {
        if let Some(value) = self.value {
            self.value = Some(MinMax::min_propagate_nan(value, other));
        } else {
            self.value = Some(other);
        }
    }
}

#[cfg(feature = "propagate_nans")]
impl<T: PolarsFloatType + Clone> Reduction for crate::reduce::extrema::MinNanReduce<T>
where
    T::Native: MinMax,
{
    fn init_dyn(&self) -> Box<dyn Reduction> {
        Box::new(Self::new())
    }
    fn reset(&mut self) {
        self.value = None;
    }

    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        if let Some(v) = nan_propagating_aggregate::ca_nan_agg(
            batch.unpack::<T>().unwrap(),
            MinMax::min_propagate_nan,
        ) {
            self.update_impl(v)
        }
        Ok(())
    }

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();

        if let Some(value) = &other.value {
            self.update_impl(*value);
        }
        Ok(())
    }

    fn finalize(&mut self) -> PolarsResult<Scalar> {
        let av = AnyValue::from(self.value);
        Ok(Scalar::new(T::get_dtype(), av))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
