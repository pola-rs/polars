use std::sync::Arc;

use arrow::datatypes::Metadata;
use arrow::io::ipc::read::{read_stream_metadata, StreamReader, StreamState};
use arrow::io::ipc::write::WriteOptions;
use polars_error::{polars_err, to_compute_err, PolarsResult};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_serialize::deserialize_map_bytes;
use polars_utils::pl_str::PlSmallStr;
use serde::de::Error;
use serde::*;

use crate::chunked_array::flags::StatisticsFlags;
use crate::config;
use crate::frame::chunk_df_for_writing;
use crate::prelude::{CompatLevel, DataFrame, SchemaExt};
use crate::utils::accumulate_dataframes_vertical_unchecked;

const FLAGS_KEY: PlSmallStr = PlSmallStr::from_static("_PL_FLAGS");

impl DataFrame {
    pub fn serialize_into_writer(&mut self, writer: &mut dyn std::io::Write) -> PolarsResult<()> {
        let schema = self.schema();

        if schema.iter_values().any(|x| x.is_object()) {
            return Err(polars_err!(
                ComputeError:
                "serializing data of type Object is not supported",
            ));
        }

        let mut ipc_writer =
            arrow::io::ipc::write::StreamWriter::new(writer, WriteOptions { compression: None });

        ipc_writer.set_custom_schema_metadata(Arc::new(Metadata::from_iter(
            self.get_columns().iter().map(|c| {
                (
                    format_pl_smallstr!("{}{}", FLAGS_KEY, c.name()),
                    PlSmallStr::from(c.get_flags().bits().to_string()),
                )
            }),
        )));

        ipc_writer.set_custom_schema_metadata(Arc::new(Metadata::from([(
            FLAGS_KEY,
            serde_json::to_string(
                &self
                    .iter()
                    .map(|s| s.get_flags().bits())
                    .collect::<Vec<u32>>(),
            )
            .map_err(to_compute_err)?
            .into(),
        )])));

        ipc_writer.start(&schema.to_arrow(CompatLevel::newest()), None)?;

        for batch in chunk_df_for_writing(self, 512 * 512)?.iter_chunks(CompatLevel::newest(), true)
        {
            ipc_writer.write(&batch, None)?;
        }

        ipc_writer.finish()?;

        Ok(())
    }

    pub fn serialize_to_bytes(&mut self) -> PolarsResult<Vec<u8>> {
        let mut buf = vec![];
        self.serialize_into_writer(&mut buf)?;

        Ok(buf)
    }

    pub fn deserialize_from_reader(reader: &mut dyn std::io::Read) -> PolarsResult<Self> {
        let mut md = read_stream_metadata(reader)?;
        let arrow_schema = md.schema.clone();

        let custom_metadata = md.custom_schema_metadata.take();

        let reader = StreamReader::new(reader, md, None);
        let dfs = reader
            .into_iter()
            .map_while(|batch| match batch {
                Ok(StreamState::Some(batch)) => Some(DataFrame::try_from((batch, &arrow_schema))),
                Ok(StreamState::Waiting) => None,
                Err(e) => Some(Err(e)),
            })
            .collect::<PolarsResult<Vec<DataFrame>>>()?;

        if dfs.is_empty() {
            return Ok(DataFrame::empty());
        }
        let mut df = accumulate_dataframes_vertical_unchecked(dfs);

        // Set custom metadata (fallible)
        (|| {
            let custom_metadata = custom_metadata?;
            let flags = custom_metadata.get(&FLAGS_KEY)?;

            let flags: PolarsResult<Vec<u32>> = serde_json::from_str(flags).map_err(to_compute_err);

            let verbose = config::verbose();

            if let Err(e) = &flags {
                if verbose {
                    eprintln!("DataFrame::read_ipc: Error parsing metadata flags: {}", e);
                }
            }

            let flags = flags.ok()?;

            if flags.len() != df.width() {
                if verbose {
                    eprintln!(
                        "DataFrame::read_ipc: Metadata flags width mismatch: {} != {}",
                        flags.len(),
                        df.width()
                    );
                }

                return None;
            }

            let mut n_set = 0;

            for (c, v) in unsafe { df.get_columns_mut() }.iter_mut().zip(flags) {
                if let Some(flags) = StatisticsFlags::from_bits(v) {
                    n_set += c.set_flags(flags) as usize;
                }
            }

            if verbose {
                eprintln!(
                    "DataFrame::read_ipc: Loaded metadata for {} / {} columns",
                    n_set,
                    df.width()
                );
            }

            Some(())
        })();

        Ok(df)
    }
}

impl Serialize for DataFrame {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error;

        let mut bytes = vec![];
        self.clone()
            .serialize_into_writer(&mut bytes)
            .map_err(S::Error::custom)?;

        serializer.serialize_bytes(bytes.as_slice())
    }
}

impl<'de> Deserialize<'de> for DataFrame {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
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
