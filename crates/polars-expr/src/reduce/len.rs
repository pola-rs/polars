use polars_core::error::constants::LENGTH_LIMIT_MSG;

use super::*;

#[derive(Clone)]
pub struct LenReduce {
    len: u64,
}

impl LenReduce {
    pub(crate) fn new() -> Self {
        Self { len: 0 }
    }
}

impl Reduction for LenReduce {
    fn init_dyn(&self) -> Box<dyn Reduction> {
        Box::new(Self::new())
    }

    fn reset(&mut self) {
        self.len = 0;
    }

    fn update(&mut self, batch: &Series) -> PolarsResult<()> {
        self.len += batch.len() as u64;
        Ok(())
    }

    fn combine(&mut self, other: &dyn Reduction) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        self.len += other.len;
        Ok(())
    }

    fn finalize(&mut self) -> PolarsResult<Scalar> {
        #[allow(clippy::useless_conversion)]
        let as_idx: IdxSize = self.len.try_into().expect(LENGTH_LIMIT_MSG);
        Ok(Scalar::new(IDX_DTYPE, as_idx.into()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
