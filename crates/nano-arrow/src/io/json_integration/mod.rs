//! Utils for JSON integration testing
//!
//! These utilities define structs that read the integration JSON format for integration testing purposes.

use serde_derive::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Error;

pub mod read;
pub mod write;

/// A struct that represents an Arrow file with a schema and record batches
#[derive(Deserialize, Serialize, Debug)]
pub struct ArrowJson {
    /// The schema
    pub schema: ArrowJsonSchema,
    /// The batches
    pub batches: Vec<ArrowJsonBatch>,
    /// The dictionaries
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dictionaries: Option<Vec<ArrowJsonDictionaryBatch>>,
}

/// A struct that partially reads the Arrow JSON schema.
///
/// Fields are left as JSON `Value` as they vary by `DataType`
#[derive(Deserialize, Serialize, Debug)]
pub struct ArrowJsonSchema {
    /// The fields
    pub fields: Vec<ArrowJsonField>,
    /// The metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// Fields are left as JSON `Value` as they vary by `DataType`
#[derive(Deserialize, Serialize, Debug)]
pub struct ArrowJsonField {
    /// The name
    pub name: String,
    /// The type
    #[serde(rename = "type")]
    pub field_type: Value,
    /// whether it is nullable
    pub nullable: bool,
    /// the children
    pub children: Vec<ArrowJsonField>,
    /// the dictionary
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dictionary: Option<ArrowJsonFieldDictionary>,
    /// the fields' metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// Dictionary metadata
#[derive(Deserialize, Serialize, Debug)]
pub struct ArrowJsonFieldDictionary {
    /// the dictionary id
    pub id: i64,
    /// the index type
    #[serde(rename = "indexType")]
    pub index_type: IntegerType,
    /// whether it is ordered
    #[serde(rename = "isOrdered")]
    pub is_ordered: bool,
}

/// the type of the integer in the dictionary
#[derive(Deserialize, Serialize, Debug)]
pub struct IntegerType {
    /// its name
    pub name: String,
    /// whether it is signed
    #[serde(rename = "isSigned")]
    pub is_signed: bool,
    /// the bit width
    #[serde(rename = "bitWidth")]
    pub bit_width: i64,
}

/// A struct that partially reads the Arrow JSON record batch
#[derive(Deserialize, Serialize, Debug)]
pub struct ArrowJsonBatch {
    count: usize,
    /// the columns
    pub columns: Vec<ArrowJsonColumn>,
}

/// A struct that partially reads the Arrow JSON dictionary batch
#[derive(Deserialize, Serialize, Debug)]
#[allow(non_snake_case)]
pub struct ArrowJsonDictionaryBatch {
    /// the id
    pub id: i64,
    /// the dictionary batch
    pub data: ArrowJsonBatch,
}

/// A struct that partially reads the Arrow JSON column/array
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ArrowJsonColumn {
    name: String,
    /// the number of elements
    pub count: usize,
    /// the validity bitmap
    #[serde(rename = "VALIDITY")]
    pub validity: Option<Vec<u8>>,
    /// the data
    #[serde(rename = "DATA")]
    pub data: Option<Vec<Value>>,
    /// the offsets
    #[serde(rename = "OFFSET")]
    pub offset: Option<Vec<Value>>, // leaving as Value as 64-bit offsets are strings
    /// the type id for union types
    #[serde(rename = "TYPE_ID")]
    pub type_id: Option<Vec<Value>>,
    /// the children
    pub children: Option<Vec<ArrowJsonColumn>>,
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Error::ExternalFormat(error.to_string())
    }
}
