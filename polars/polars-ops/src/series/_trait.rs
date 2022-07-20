use super::*;
#[cfg(feature = "hash")]
use polars_core::export::ahash;
use std::ops::Deref;

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

#[cfg(feature = "hash")]
macro_rules! invalid_operation_panic {
    ($s:expr) => {
        panic!(
            "this operation is not implemented/valid for this dtype: {:?}",
            $s.dtype()
        )
    };
}

pub trait SeriesOps {
    fn dtype(&self) -> &DataType;

    #[cfg(feature = "to_dummies")]
    fn to_dummies(&self) -> Result<DataFrame> {
        invalid_operation!(self)
    }

    #[cfg(feature = "hash")]
    fn hash(&self, _build_hasher: ahash::RandomState) -> UInt64Chunked {
        invalid_operation_panic!(self)
    }
}

impl SeriesOps for Series {
    fn dtype(&self) -> &DataType {
        self.deref().dtype()
    }
    #[cfg(feature = "to_dummies")]
    fn to_dummies(&self) -> Result<DataFrame> {
        self.to_ops().to_dummies()
    }

    #[cfg(feature = "hash")]
    fn hash(&self, build_hasher: ahash::RandomState) -> UInt64Chunked {
        match self.dtype() {
            DataType::List(_) => {
                let ca = self.list().unwrap();
                crate::chunked_array::hash::hash(ca, build_hasher)
            }
            _ => UInt64Chunked::from_vec(self.name(), self.0.vec_hash(build_hasher)),
        }
    }
}
