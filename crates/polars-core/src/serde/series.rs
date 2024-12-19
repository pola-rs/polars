use std::fmt::Formatter;

use arrow::datatypes::Metadata;
use arrow::io::ipc::read::{read_stream_metadata, StreamReader, StreamState};
use arrow::io::ipc::write::WriteOptions;
use serde::de::{Error as DeError, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::chunked_array::flags::StatisticsFlags;
use crate::config;
use crate::prelude::*;
use crate::utils::accumulate_dataframes_vertical;

const FLAGS_KEY: PlSmallStr = PlSmallStr::from_static("_PL_FLAGS");

impl Serialize for Series {
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error;

        if self.dtype().is_object() {
            return Err(polars_err!(
                ComputeError:
                "serializing data of type Object is not supported",
            ))
            .map_err(S::Error::custom);
        }

        let bytes = vec![];
        let mut bytes = std::io::Cursor::new(bytes);
        let mut ipc_writer = arrow::io::ipc::write::StreamWriter::new(
            &mut bytes,
            WriteOptions {
                // Compression should be done on an outer level
                compression: Some(arrow::io::ipc::write::Compression::ZSTD),
            },
        );

        let df = unsafe {
            DataFrame::new_no_checks_height_from_first(vec![self.rechunk().into_column()])
        };

        ipc_writer.set_custom_schema_metadata(Arc::new(Metadata::from([(
            FLAGS_KEY,
            PlSmallStr::from(self.get_flags().bits().to_string()),
        )])));

        ipc_writer
            .start(
                &ArrowSchema::from_iter([Field {
                    name: self.name().clone(),
                    dtype: self.dtype().clone(),
                }
                .to_arrow(CompatLevel::newest())]),
                None,
            )
            .map_err(S::Error::custom)?;

        for batch in df.iter_chunks(CompatLevel::newest(), false) {
            ipc_writer.write(&batch, None).map_err(S::Error::custom)?;
        }

        ipc_writer.finish().map_err(S::Error::custom)?;
        serializer.serialize_bytes(bytes.into_inner().as_slice())
    }
}

impl<'de> Deserialize<'de> for Series {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, <D as Deserializer<'de>>::Error>
    where
        D: Deserializer<'de>,
    {
        struct SeriesVisitor;

        impl<'de> Visitor<'de> for SeriesVisitor {
            type Value = Series;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                formatter.write_str("bytes (IPC)")
            }

            fn visit_bytes<E>(self, mut v: &[u8]) -> Result<Self::Value, E>
            where
                E: DeError,
            {
                let mut md = read_stream_metadata(&mut v).map_err(E::custom)?;
                let arrow_schema = md.schema.clone();

                let custom_metadata = md.custom_schema_metadata.take();

                let reader = StreamReader::new(v, md, None);
                let dfs = reader
                    .into_iter()
                    .map_while(|batch| match batch {
                        Ok(StreamState::Some(batch)) => {
                            Some(DataFrame::try_from((batch, &arrow_schema)))
                        },
                        Ok(StreamState::Waiting) => None,
                        Err(e) => Some(Err(e)),
                    })
                    .collect::<PolarsResult<Vec<DataFrame>>>()
                    .map_err(E::custom)?;

                let df = accumulate_dataframes_vertical(dfs).map_err(E::custom)?;

                if df.width() != 1 {
                    return Err(polars_err!(
                        ShapeMismatch:
                        "expected only 1 column when deserializing Series from IPC, got columns: {:?}",
                        df.schema().iter_names().collect::<Vec<_>>()
                    )).map_err(E::custom);
                }

                let mut s = df.take_columns().swap_remove(0).take_materialized_series();

                if let Some(custom_metadata) = custom_metadata {
                    if let Some(flags) = custom_metadata.get(&FLAGS_KEY) {
                        if let Ok(v) = flags.parse::<u32>() {
                            if let Some(flags) = StatisticsFlags::from_bits(v) {
                                s.set_flags(flags);
                            }
                        } else if config::verbose() {
                            eprintln!("Series::Deserialize: Failed to parse as u8: {:?}", flags)
                        }
                    }
                }

                Ok(s)
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                // This is not ideal, but we hit here if the serialization format is JSON.
                let bytes = std::iter::from_fn(|| seq.next_element::<u8>().transpose())
                    .collect::<Result<Vec<_>, A::Error>>()?;

                self.visit_bytes(&bytes)
            }
        }

        deserializer.deserialize_bytes(SeriesVisitor)
    }
}
