use std::ops::Deref;

use super::*;

#[cfg(feature = "to_dummies")]
macro_rules! invalid_operation {
    ($s:expr) => {
        Err(PolarsError::InvalidOperation(
            format!(
                "this operation is not implemented/valid for this dtype: {:?}",
                $s.dtype()
            )
            .into(),
        ))
    };
}

pub trait SeriesOps {
    fn dtype(&self) -> &DataType;

    #[cfg(feature = "to_dummies")]
    fn to_dummies(&self) -> PolarsResult<DataFrame> {
        invalid_operation!(self)
    }
}

impl SeriesOps for Series {
    fn dtype(&self) -> &DataType {
        self.deref().dtype()
    }
    #[cfg(feature = "to_dummies")]
    fn to_dummies(&self) -> PolarsResult<DataFrame> {
        self.to_ops().to_dummies()
    }
}
