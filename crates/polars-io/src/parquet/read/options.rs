use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use polars_core::schema::SchemaRef;
use polars_parquet::parquet::encryption::decrypt::FileDecryptionProperties;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct ParquetOptions {
    pub schema: Option<SchemaRef>,
    pub parallel: ParallelStrategy,
    pub low_memory: bool,
    pub use_statistics: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    pub decryption_properties: Option<PlFileDecryptionProperties>,
}

impl Default for ParquetOptions {
    fn default() -> Self {
        Self {
            schema: None,
            parallel: ParallelStrategy::default(),
            low_memory: false,
            use_statistics: true,
            decryption_properties: None,
        }
    }
}

/// Properties for reading files encrypted with Parquet modular encryption.
///
/// These hold secret keys, so can't be serialized, and are compared and hashed
/// by pointer.
#[derive(Clone)]
pub struct PlFileDecryptionProperties(pub Arc<FileDecryptionProperties>);

impl Debug for PlFileDecryptionProperties {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // FileDecryptionProperties doesn't output keys.
        self.0.fmt(f)
    }
}

impl Eq for PlFileDecryptionProperties {}

impl PartialEq for PlFileDecryptionProperties {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Hash for PlFileDecryptionProperties {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(Arc::as_ptr(&self.0) as usize)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for PlFileDecryptionProperties {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        Err(D::Error::custom(
            "cannot deserialize parquet decryption properties",
        ))
    }
}

#[cfg(feature = "serde")]
impl Serialize for PlFileDecryptionProperties {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;
        Err(S::Error::custom(
            "cannot serialize parquet decryption properties",
        ))
    }
}

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for PlFileDecryptionProperties {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "PlFileDecryptionProperties".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "PlFileDecryptionProperties"))
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Vec::<u8>::json_schema(generator)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ParallelStrategy {
    /// Don't parallelize
    None,
    /// Parallelize over the columns
    Columns,
    /// Parallelize over the row groups
    RowGroups,
    /// First evaluates the pushed-down predicates in parallel and determines a mask of which rows
    /// to read. Then, it parallelizes over both the columns and the row groups while filtering out
    /// rows that do not need to be read. This can provide significant speedups for large files
    /// (i.e. many row-groups) with a predicate that filters clustered rows or filters heavily. In
    /// other cases, this may slow down the scan compared other strategies.
    ///
    /// If no predicate is given, this falls back to back to [`ParallelStrategy::Auto`].
    Prefiltered,
    /// Automatically determine over which unit to parallelize
    /// This will choose the most occurring unit.
    #[default]
    Auto,
}
