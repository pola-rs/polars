use polars_utils::pl_serialize::deserialize_map_bytes;
use serde::de::Error;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::prelude::*;

impl Series {
    pub fn serialize_into_writer(&self, writer: &mut dyn std::io::Write) -> PolarsResult<()> {
        let mut df =
            unsafe { DataFrame::new_no_checks_height_from_first(vec![self.clone().into_column()]) };

        df.serialize_into_writer(writer)
    }

    pub fn serialize_to_bytes(&self) -> PolarsResult<Vec<u8>> {
        let mut buf = vec![];
        self.serialize_into_writer(&mut buf)?;

        Ok(buf)
    }

    pub fn deserialize_from_reader(reader: &mut dyn std::io::Read) -> PolarsResult<Self> {
        let df = DataFrame::deserialize_from_reader(reader)?;

        if df.width() != 1 {
            polars_bail!(
                ShapeMismatch:
                "expected only 1 column when deserializing Series from IPC, got columns: {:?}",
                df.schema().iter_names().collect::<Vec<_>>()
            )
        }

        Ok(df.take_columns().swap_remove(0).take_materialized_series())
    }
}

impl Serialize for Series {
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error;

        serializer.serialize_bytes(
            self.serialize_to_bytes()
                .map_err(S::Error::custom)?
                .as_slice(),
        )
    }
}

impl<'de> Deserialize<'de> for Series {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, <D as Deserializer<'de>>::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_map_bytes(deserializer, &mut |b| {
            let v = &mut b.as_ref();
            Self::deserialize_from_reader(v)
        })?
        .map_err(D::Error::custom)
    }
}
