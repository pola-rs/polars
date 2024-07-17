use polars_core::prelude::{AnyValue, DataType};

use super::*;

#[derive(Clone)]
pub struct SumReduce {
    value: Scalar,
}

impl SumReduce {
    pub(crate) fn new(dtype: DataType) -> Self {
        let value = Scalar::new(dtype, AnyValue::Null);
        Self { value }
    }

    fn update_impl(&mut self, value: &AnyValue<'static>) {
        self.value.update(self.value.value().add(value))
    }
}

impl Reduction for SumReduce {
    fn init_dyn(&self) -> Box<dyn Reduction> {
        Box::new(Self::new(self.value.dtype().clone()))
    }
    fn reset(&mut self) {
        let av = AnyValue::zero(self.value.dtype());
        self.value.update(av);
    }

    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        let sc = batch.sum_reduce()?;
        self.update_impl(sc.value());
        Ok(())
    }

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        self.update_impl(other.value.value());
        Ok(())
    }

    fn finalize(&mut self) -> PolarsResult<Scalar> {
        Ok(self.value.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
