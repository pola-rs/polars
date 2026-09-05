//! Carrying the categories of an [`Enum`](crate::DataType::Enum) through serde.
//!
//! The categories are a [`PlUtf8ViewArray`], which has no serde of its own — an array is buffers,
//! and what those mean is the Arrow IPC format's business rather than serde's. So the array is
//! written as one IPC stream of a single column and handed to serde as the bytes of it, which is
//! how a `Series` reaches serde too.

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::Array;
use arrow::datatypes::{ArrowSchema, ArrowSchemaRef, Field as ArrowField};
use arrow::io::ipc::read::{StreamReader, StreamState, read_stream_metadata};
use arrow::io::ipc::write::{StreamWriter, WriteOptions};
use arrow::record_batch::RecordBatchT;
use polars_array::PlUtf8ViewArray;
use polars_array::arrow::{export, import};
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_utils::pl_str::PlSmallStr;
use serde::de::Error as _;
use serde::ser::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The name the one column of the stream is written under. It is read back positionally, so this
/// only has to stay the same for a stream to be recognisable, not to be understood.
const COLUMN_NAME: PlSmallStr = PlSmallStr::from_static("categories");

/// The categories of an enum, written through serde as one Arrow IPC stream.
#[derive(Clone)]
pub struct SerializableCategories(pub PlUtf8ViewArray);

fn schema() -> ArrowSchemaRef {
    Arc::new(ArrowSchema::from_iter([ArrowField::new(
        COLUMN_NAME,
        arrow::datatypes::ArrowDataType::Utf8View,
        false,
    )]))
}

impl SerializableCategories {
    fn to_ipc_bytes(&self) -> PolarsResult<Vec<u8>> {
        let array = export::utf8view_to_arrow_utf8view(&self.0);
        let schema = schema();

        let mut buf = Vec::new();
        let mut writer = StreamWriter::new(&mut buf, WriteOptions { compression: None });
        writer.start(&schema, None)?;
        // An enum with no categories writes no batch at all, which reads back as no categories.
        if !array.is_empty() {
            let batch = RecordBatchT::new(array.len(), schema, vec![array.boxed()]);
            writer.write(&batch, None)?;
        }
        writer.finish()?;
        Ok(buf)
    }

    fn from_ipc_bytes(bytes: &[u8]) -> PolarsResult<Self> {
        let mut reader = Cursor::new(bytes);
        let metadata = read_stream_metadata(&mut reader)?;
        let reader = StreamReader::new(&mut reader, metadata, None);

        let mut categories: Option<PlUtf8ViewArray> = None;
        for batch in reader {
            let StreamState::Some(batch) = batch? else {
                break;
            };
            let [array] = batch.arrays() else {
                polars_bail!(
                    ShapeMismatch:
                    "expected exactly one column of enum categories, got {}",
                    batch.arrays().len(),
                );
            };
            let array = import::from_arrow(array.as_ref());
            let array = array
                .as_any()
                .downcast_ref::<PlUtf8ViewArray>()
                .ok_or_else(
                    || polars_err!(SchemaMismatch: "enum categories are not a string array"),
                )?
                .clone();

            // The categories are written as one batch, so a second one is a stream this did not
            // write.
            polars_ensure!(
                categories.is_none(),
                ShapeMismatch: "enum categories arrived in more than one batch",
            );
            categories = Some(array);
        }

        Ok(Self(categories.unwrap_or_else(PlUtf8ViewArray::new_empty)))
    }
}

impl Serialize for SerializableCategories {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.to_ipc_bytes().map_err(S::Error::custom)?)
    }
}

impl<'de> Deserialize<'de> for SerializableCategories {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes = polars_utils::pl_serialize::deserialize_map_bytes(deserializer, |bytes| {
            Self::from_ipc_bytes(bytes.as_ref())
        })?;
        bytes.map_err(D::Error::custom)
    }
}

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for SerializableCategories {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SerializableCategories".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "SerializableCategories"))
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        // The stream reaches serde as its bytes, so that is what a schema can say about it.
        Vec::<u8>::json_schema(generator)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(strings: &[&str]) {
        let array = import::utf8_view_from_arrow(
            &arrow::array::Utf8ViewArray::from_slice_values(strings),
        );
        let categories = SerializableCategories(array);
        let bytes = categories.to_ipc_bytes().unwrap();
        let read = SerializableCategories::from_ipc_bytes(&bytes).unwrap();

        let read: Vec<_> = read.0.values_iter().collect();
        assert_eq!(read, strings);
    }

    /// The categories come back in the order they went out, which is the order that gives each of
    /// them its category id.
    #[test]
    fn categories_round_trip_in_order() {
        round_trip(&["a", "b", "c"]);
        round_trip(&["", "one", "two words", "\u{1F600}"]);
    }

    /// An enum with no categories at all is a stream with no batch in it, not a broken one.
    #[test]
    fn no_categories_round_trip() {
        round_trip(&[]);
    }
}
