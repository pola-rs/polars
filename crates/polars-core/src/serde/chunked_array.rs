use serde::{Serialize, Serializer};

use crate::prelude::*;

// We don't use this internally (we call Series::serialize instead), but Rust users might need it.
impl<T> Serialize for ChunkedArray<T>
where
    T: PolarsDataType,
    ChunkedArray<T>: IntoSeries,
{
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        self.clone().into_series().serialize(serializer)
    }
}
