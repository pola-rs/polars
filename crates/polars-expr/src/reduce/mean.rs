use polars_core::prelude::{AnyValue, DataType};
use polars_core::utils::Container;
use super::*;

pub struct MeanReduce {
    value: Scalar,
    len: u64,
}

impl MeanReduce {
    pub(crate) fn new(dtype: DataType) -> Self {
        let value = Scalar::new(dtype, AnyValue::Null);
        Self { value, len: 0 }
    }

    fn update_impl(&mut self, value: &AnyValue<'static>) {
        self.value.update(self.value.value().add(value))
    }
}

impl Reduction for MeanReduce {
    fn init(&mut self) {
        let av = AnyValue::zero(self.value.dtype());
        self.value.update(av);
    }

    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        let sc = batch.sum_reduce()?;
        self.update_impl(sc.value());
        self.len += batch.len() as u64;
        Ok(())
    }

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        self.update_impl(&other.value.value());
        self.len += other.len;
        Ok(())
    }

    fn finalize(&mut self) -> PolarsResult<Scalar> {
        Ok(self.value.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
