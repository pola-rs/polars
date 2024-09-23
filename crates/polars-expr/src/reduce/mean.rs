use super::*;

#[derive(Clone)]
pub struct MeanReduce {
    dtype: DataType,
}

impl MeanReduce {
    pub fn new(dtype: DataType) -> Self {
        Self { dtype }
    }
}

impl Reduction for MeanReduce {
    fn new_reducer(&self) -> Box<dyn ReductionState> {
        Box::new(MeanReduceState {
            dtype: self.dtype.clone(),
            sum: 0.0,
            count: 0,
        })
    }
}

pub struct MeanReduceState {
    dtype: DataType,
    sum: f64,
    count: u64,
}

impl ReductionState for MeanReduceState {
    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        let count = batch.len() as u64 - batch.null_count() as u64;
        self.count += count;
        self.sum += batch._sum_as_f64();
        Ok(())
    }

    fn combine(&mut self, other: &dyn ReductionState) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        self.sum += other.sum;
        self.count += other.count;
        Ok(())
    }

    fn finalize(&self) -> PolarsResult<Scalar> {
        let val = (self.count > 0).then(|| self.sum / self.count as f64);
        Ok(polars_core::scalar::reduce::mean_reduce(
            val,
            self.dtype.clone(),
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
