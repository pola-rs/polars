use serde::{Serialize, Serializer};

use crate::prelude::*;

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
