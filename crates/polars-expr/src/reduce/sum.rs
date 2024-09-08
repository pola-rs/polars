use polars_core::prelude::{AnyValue, DataType};

use super::*;

#[derive(Clone)]
pub struct SumReduce {
    dtype: DataType,
}

impl SumReduce {
    pub fn new(dtype: DataType) -> Self {
        // We cast small dtypes up in the sum, we must also do this when
        // returning the empty sum to be consistent.
        use DataType::*;
        let dtype = match dtype {
            Boolean => IDX_DTYPE,
            Int8 | UInt8 | Int16 | UInt16 => Int64,
            dt => dt,
        };
        Self { dtype }
    }
}

impl Reduction for SumReduce {
    fn new_reducer(&self) -> Box<dyn ReductionState> {
        let value = Scalar::new(self.dtype.clone(), AnyValue::zero_sum(&self.dtype));
        Box::new(SumReduceState { value })
    }
}

struct SumReduceState {
    value: Scalar,
}

impl SumReduceState {
    fn add_value(&mut self, other: &AnyValue<'_>) {
        self.value.update(self.value.value().add(other));
    }
}

impl ReductionState for SumReduceState {
    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        let reduced = batch.sum_reduce()?;
        self.add_value(reduced.value());
        Ok(())
    }

    fn combine(&mut self, other: &dyn ReductionState) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        self.add_value(other.value.value());
        Ok(())
    }

    fn finalize(&self) -> PolarsResult<Scalar> {
        Ok(self.value.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
