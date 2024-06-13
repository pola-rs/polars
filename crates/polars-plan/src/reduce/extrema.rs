use polars_core::prelude::AnyValue;

use super::*;

struct MinReduce {
    value: Scalar,
}

impl MinReduce {
    fn update_impl(&mut self, value: &AnyValue<'static>) {
        if value < self.value.value() {
            self.value.update(value.clone());
        }
    }
}

impl Reduction for MinReduce {
    fn init(&mut self) {
        let av = AnyValue::zero(self.value.dtype());
        self.value.update(av);
    }

    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        let sc = batch.min_reduce()?;
        self.update_impl(sc.value());
        Ok(())
    }

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        self.update_impl(&other.value.value());
        Ok(())
    }

    fn finalize(&mut self) -> PolarsResult<Scalar> {
        Ok(self.value.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
struct MaxReduce {
    value: Scalar,
}

impl MaxReduce {
    fn update_impl(&mut self, value: &AnyValue<'static>) {
        if value > self.value.value() {
            self.value.update(value.clone());
        }
    }
}

impl Reduction for MaxReduce {
    fn init(&mut self) {
        let av = AnyValue::zero(self.value.dtype());
        self.value.update(av);
    }

    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        let sc = batch.max_reduce()?;
        self.update_impl(sc.value());
        Ok(())
    }

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        self.update_impl(&other.value.value());
        Ok(())
    }

    fn finalize(&mut self) -> PolarsResult<Scalar> {
        Ok(self.value.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
