use polars_core::error::constants::LENGTH_LIMIT_MSG;

use super::*;

#[derive(Clone)]
pub struct LenReduce {}

impl LenReduce {
    pub fn new() -> Self {
        Self {}
    }
}

impl Reduction for LenReduce {
    fn new_reducer(&self) -> Box<dyn ReductionState> {
        Box::new(LenReduceState { len: 0 })
    }
}

pub struct LenReduceState {
    len: u64,
}

impl ReductionState for LenReduceState {
    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        self.len += batch.len() as u64;
        Ok(())
    }

    fn combine(&mut self, other: &dyn ReductionState) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        self.len += other.len;
        Ok(())
    }

    fn finalize(&self) -> PolarsResult<Scalar> {
        #[allow(clippy::useless_conversion)]
        let as_idx: IdxSize = self.len.try_into().expect(LENGTH_LIMIT_MSG);
        Ok(Scalar::new(IDX_DTYPE, as_idx.into()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
