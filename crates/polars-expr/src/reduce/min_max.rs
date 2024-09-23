use super::*;

#[derive(Clone)]
pub struct MinReduce {
    dtype: DataType,
}

impl MinReduce {
    pub fn new(dtype: DataType) -> Self {
        Self { dtype }
    }
}

impl Reduction for MinReduce {
    fn new_reducer(&self) -> Box<dyn ReductionState> {
        Box::new(MinReduceState {
            value: Scalar::new(self.dtype.clone(), AnyValue::Null),
        })
    }
}

struct MinReduceState {
    value: Scalar,
}

impl MinReduceState {
    fn update_with_value(&mut self, other: &AnyValue<'static>) {
        if self.value.is_null()
            || !other.is_null() && (other < self.value.value() || self.value.is_nan())
        {
            self.value.update(other.clone());
        }
    }
}

impl ReductionState for MinReduceState {
    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        let sc = batch.min_reduce()?;
        self.update_with_value(sc.value());
        Ok(())
    }

    fn combine(&mut self, other: &dyn ReductionState) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        self.update_with_value(other.value.value());
        Ok(())
    }

    fn finalize(&self) -> PolarsResult<Scalar> {
        Ok(self.value.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Clone)]
pub struct MaxReduce {
    dtype: DataType,
}

impl MaxReduce {
    pub fn new(dtype: DataType) -> Self {
        Self { dtype }
    }
}

impl Reduction for MaxReduce {
    fn new_reducer(&self) -> Box<dyn ReductionState> {
        Box::new(MaxReduceState {
            value: Scalar::new(self.dtype.clone(), AnyValue::Null),
        })
    }
}

struct MaxReduceState {
    value: Scalar,
}

impl MaxReduceState {
    fn update_with_value(&mut self, other: &AnyValue<'static>) {
        if self.value.is_null()
            || !other.is_null() && (other > self.value.value() || self.value.is_nan())
        {
            self.value.update(other.clone());
        }
    }
}

impl ReductionState for MaxReduceState {
    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        let sc = batch.min_reduce()?;
        self.update_with_value(sc.value());
        Ok(())
    }

    fn combine(&mut self, other: &dyn ReductionState) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        self.update_with_value(other.value.value());
        Ok(())
    }

    fn finalize(&self) -> PolarsResult<Scalar> {
        Ok(self.value.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
