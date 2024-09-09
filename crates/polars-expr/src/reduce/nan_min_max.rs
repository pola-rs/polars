use std::marker::PhantomData;

use polars_compute::min_max::MinMaxKernel;
use polars_core::datatypes::PolarsFloatType;
use polars_utils::min_max::MinMax;

use super::*;

#[derive(Clone)]
pub struct NanMinReduce<F> {
    _phantom: PhantomData<F>,
}

impl<F> NanMinReduce<F> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F: PolarsFloatType> Reduction for NanMinReduce<F>
where
    F::Array: for<'a> MinMaxKernel<Scalar<'a> = F::Native>,
{
    fn new_reducer(&self) -> Box<dyn ReductionState> {
        Box::new(NanMinReduceState::<F> { value: None })
    }
}

struct NanMinReduceState<F: PolarsFloatType> {
    value: Option<F::Native>,
}

impl<F: PolarsFloatType> NanMinReduceState<F> {
    fn update_with_value(&mut self, other: Option<F::Native>) {
        if let Some(other) = other {
            if let Some(value) = self.value {
                self.value = Some(MinMax::min_propagate_nan(value, other));
            } else {
                self.value = Some(other);
            }
        }
    }
}

impl<F: PolarsFloatType> ReductionState for NanMinReduceState<F>
where
    F::Array: for<'a> MinMaxKernel<Scalar<'a> = F::Native>,
{
    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        let ca = batch.unpack::<F>().unwrap();
        let reduced = ca
            .downcast_iter()
            .filter_map(MinMaxKernel::min_propagate_nan_kernel)
            .reduce(MinMax::min_propagate_nan);
        self.update_with_value(reduced);
        Ok(())
    }

    fn combine(&mut self, other: &dyn ReductionState) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        self.update_with_value(other.value);
        Ok(())
    }

    fn finalize(&self) -> PolarsResult<Scalar> {
        Ok(Scalar::new(F::get_dtype(), AnyValue::from(self.value)))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Clone)]
pub struct NanMaxReduce<F> {
    _phantom: PhantomData<F>,
}

impl<F> NanMaxReduce<F> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F: PolarsFloatType> Reduction for NanMaxReduce<F>
where
    F::Array: for<'a> MinMaxKernel<Scalar<'a> = F::Native>,
{
    fn new_reducer(&self) -> Box<dyn ReductionState> {
        Box::new(NanMaxReduceState::<F> { value: None })
    }
}

struct NanMaxReduceState<F: PolarsFloatType> {
    value: Option<F::Native>,
}

impl<F: PolarsFloatType> NanMaxReduceState<F> {
    fn update_with_value(&mut self, other: Option<F::Native>) {
        if let Some(other) = other {
            if let Some(value) = self.value {
                self.value = Some(MinMax::max_propagate_nan(value, other));
            } else {
                self.value = Some(other);
            }
        }
    }
}

impl<F: PolarsFloatType> ReductionState for NanMaxReduceState<F>
where
    F::Array: for<'a> MinMaxKernel<Scalar<'a> = F::Native>,
{
    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        let ca = batch.unpack::<F>().unwrap();
        let reduced = ca
            .downcast_iter()
            .filter_map(MinMaxKernel::max_propagate_nan_kernel)
            .reduce(MinMax::max_propagate_nan);
        self.update_with_value(reduced);
        Ok(())
    }

    fn combine(&mut self, other: &dyn ReductionState) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        self.update_with_value(other.value);
        Ok(())
    }

    fn finalize(&self) -> PolarsResult<Scalar> {
        Ok(Scalar::new(F::get_dtype(), AnyValue::from(self.value)))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
