use serde::{Serialize, Serializer};

use crate::prelude::*;

// We don't use this internally (we call Series::serialize instead), but Rust users might need it.
impl<T: PolarsPhysicalType> Serialize for ChunkedArray<T> {
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        self.clone().into_series().serialize(serializer)
    }
}
