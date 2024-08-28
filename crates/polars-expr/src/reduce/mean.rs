use super::*;

#[derive(Clone)]
pub struct MeanReduce {
    value: Option<f64>,
    len: u64,
    dtype: DataType,
}

impl MeanReduce {
    pub(crate) fn new(dtype: DataType) -> Self {
        let value = None;
        Self {
            value,
            len: 0,
            dtype,
        }
    }

    fn update_impl(&mut self, value: &AnyValue<'static>, len: u64) {
        let value = value.extract::<f64>().expect("phys numeric");
        if let Some(acc) = &mut self.value {
            *acc += value;
            self.len += len;
        } else {
            self.value = Some(value);
            self.len = len;
        }
    }
}

impl Reduction for MeanReduce {
    fn init_dyn(&self) -> Box<dyn Reduction> {
        Box::new(Self::new(self.dtype.clone()))
    }
    fn reset(&mut self) {
        self.value = None;
        self.len = 0;
    }

    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        let sc = batch.sum_reduce()?;
        self.update_impl(sc.value(), batch.len() as u64);
        Ok(())
    }

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();

        match (self.value, other.value) {
            (Some(l), Some(r)) => self.value = Some(l + r),
            (None, Some(r)) => self.value = Some(r),
            (Some(l), None) => self.value = Some(l),
            (None, None) => self.value = None,
        }
        self.len += other.len;
        Ok(())
    }

    fn finalize(&mut self) -> PolarsResult<Scalar> {
        Ok(polars_core::scalar::reduce::mean_reduce(
            self.value.map(|v| v / self.len as f64),
            self.dtype.clone(),
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
