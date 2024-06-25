use polars_utils::IdxSize;

use super::{Metadata, MetadataFlags};
use crate::chunked_array::{IntoScalar, PolarsDataType, Scalar};

pub trait MetadataTrait {
    fn get_flags(&self) -> MetadataFlags;
    fn min_value(&self) -> Option<Scalar>;
    fn max_value(&self) -> Option<Scalar>;

    /// Number of unique non-null values
    fn distinct_count(&self) -> Option<IdxSize>;
}

impl<T: PolarsDataType> MetadataTrait for Metadata<T>
where
    T::OwnedPhysical: IntoScalar + Clone,
{
    fn get_flags(&self) -> MetadataFlags {
        self.get_flags()
    }

    fn min_value(&self) -> Option<Scalar> {
        self.get_min_value()
            .map(|v| v.clone().into_scalar(T::get_dtype()).unwrap())
    }

    fn max_value(&self) -> Option<Scalar> {
        self.get_max_value()
            .map(|v| v.clone().into_scalar(T::get_dtype()).unwrap())
    }

    fn distinct_count(&self) -> Option<IdxSize> {
        self.get_distinct_count()
    }
}
