use polars_core::prelude::PlHashMap;
use polars_utils::pl_str::PlSmallStr;

#[derive(Debug, serde::Deserialize)]
pub struct CatalogInfo {
    pub name: String,

    pub comment: Option<String>,

    #[serde(default)]
    pub storage_location: Option<String>,

    #[serde(default, deserialize_with = "null_to_default")]
    pub properties: PlHashMap<PlSmallStr, String>,

    #[serde(default, deserialize_with = "null_to_default")]
    pub options: PlHashMap<PlSmallStr, String>,

    #[serde(with = "chrono::serde::ts_milliseconds_option")]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,

    pub created_by: Option<String>,

    #[serde(with = "chrono::serde::ts_milliseconds_option")]
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,

    pub updated_by: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
pub struct NamespaceInfo {
    pub name: String,
    pub comment: Option<String>,

    #[serde(default, deserialize_with = "null_to_default")]
    pub properties: PlHashMap<PlSmallStr, String>,

    #[serde(default)]
    pub storage_location: Option<String>,

    #[serde(with = "chrono::serde::ts_milliseconds_option")]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,

    pub created_by: Option<String>,

    #[serde(with = "chrono::serde::ts_milliseconds_option")]
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,

    pub updated_by: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
pub struct TableInfo {
    pub name: String,
    pub table_id: String,
    pub table_type: TableType,

    #[serde(default)]
    pub comment: Option<String>,

    #[serde(default)]
    pub storage_location: Option<String>,

    #[serde(default)]
    pub data_source_format: Option<DataSourceFormat>,

    #[serde(default)]
    pub columns: Option<Vec<ColumnInfo>>,

    #[serde(default, deserialize_with = "null_to_default")]
    pub properties: PlHashMap<PlSmallStr, String>,

    #[serde(with = "chrono::serde::ts_milliseconds_option")]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,

    pub created_by: Option<String>,

    #[serde(with = "chrono::serde::ts_milliseconds_option")]
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,

    pub updated_by: Option<String>,
}

#[derive(
    Debug, strum_macros::Display, strum_macros::EnumString, serde::Serialize, serde::Deserialize,
)]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TableType {
    Managed,
    External,
    View,
    MaterializedView,
    StreamingTable,
    ManagedShallowClone,
    Foreign,
    ExternalShallowClone,
}

#[derive(
    Debug, strum_macros::Display, strum_macros::EnumString, serde::Serialize, serde::Deserialize,
)]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DataSourceFormat {
    Delta,
    Csv,
    Json,
    Avro,
    Parquet,
    Orc,
    Text,

    // Databricks-specific
    UnityCatalog,
    Deltasharing,
    DatabricksFormat,
    MysqlFormat,
    PostgresqlFormat,
    RedshiftFormat,
    SnowflakeFormat,
    SqldwFormat,
    SqlserverFormat,
    SalesforceFormat,
    BigqueryFormat,
    NetsuiteFormat,
    WorkdayRaasFormat,
    HiveSerde,
    HiveCustom,
    VectorIndexFormat,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ColumnInfo {
    pub name: PlSmallStr,
    pub type_name: PlSmallStr,
    pub type_text: PlSmallStr,
    pub type_json: String,
    pub position: Option<u32>,
    pub comment: Option<String>,
    pub partition_index: Option<u32>,
}

/// Note: This struct contains all the field names for a few different possible type / field presence
/// combinations. We use serde(default) and skip_serializing_if to get the desired serialization
/// output.
///
/// E.g.:
///
/// ```text
/// {
///     "name": "List",
///     "type": {"type": "array", "elementType": "long", "containsNull": True},
///     "nullable": True,
///     "metadata": {},
/// }
/// {
///     "name": "Struct",
///     "type": {
///         "type": "struct",
///         "fields": [{"name": "x", "type": "long", "nullable": True, "metadata": {}}],
///     },
///     "nullable": True,
///     "metadata": {},
/// }
/// {
///     "name": "ListStruct",
///     "type": {
///         "type": "array",
///         "elementType": {
///             "type": "struct",
///             "fields": [{"name": "x", "type": "long", "nullable": True, "metadata": {}}],
///         },
///         "containsNull": True,
///     },
///     "nullable": True,
///     "metadata": {},
/// }
/// {
///     "name": "Map",
///     "type": {
///         "type": "map",
///         "keyType": "string",
///         "valueType": "string",
///         "valueContainsNull": True,
///     },
///     "nullable": True,
///     "metadata": {},
/// }
/// ```
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ColumnTypeJson {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<PlSmallStr>,

    #[serde(rename = "type")]
    pub type_: ColumnTypeJsonType,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nullable: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<PlHashMap<String, String>>,

    // Used for List types
    #[serde(
        default,
        rename = "elementType",
        skip_serializing_if = "Option::is_none"
    )]
    pub element_type: Option<ColumnTypeJsonType>,

    #[serde(
        default,
        rename = "containsNull",
        skip_serializing_if = "Option::is_none"
    )]
    pub contains_null: Option<bool>,

    // Used for Struct types
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fields: Option<Vec<ColumnTypeJson>>,

    // Used for Map types
    #[serde(default, rename = "keyType", skip_serializing_if = "Option::is_none")]
    pub key_type: Option<ColumnTypeJsonType>,

    #[serde(default, rename = "valueType", skip_serializing_if = "Option::is_none")]
    pub value_type: Option<ColumnTypeJsonType>,

    #[serde(
        default,
        rename = "valueContainsNull",
        skip_serializing_if = "Option::is_none"
    )]
    pub value_contains_null: Option<bool>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum ColumnTypeJsonType {
    /// * `{"type": "name", ..}``
    TypeName(PlSmallStr),
    /// * `{"type": {"type": "name", ..}}`
    TypeJson(Box<ColumnTypeJson>),
}

impl Default for ColumnTypeJsonType {
    fn default() -> Self {
        Self::TypeName(PlSmallStr::EMPTY)
    }
}

impl ColumnTypeJsonType {
    pub const fn from_static_type_name(type_name: &'static str) -> Self {
        Self::TypeName(PlSmallStr::from_static(type_name))
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct TableCredentials {
    pub aws_temp_credentials: Option<TableCredentialsAws>,
    pub azure_user_delegation_sas: Option<TableCredentialsAzure>,
    pub gcp_oauth_token: Option<TableCredentialsGcp>,
    pub expiration_time: i64,
}

impl TableCredentials {
    pub fn into_enum(self) -> Option<TableCredentialsVariants> {
        if let v @ Some(_) = self.aws_temp_credentials {
            v.map(TableCredentialsVariants::Aws)
        } else if let v @ Some(_) = self.azure_user_delegation_sas {
            v.map(TableCredentialsVariants::Azure)
        } else if let v @ Some(_) = self.gcp_oauth_token {
            v.map(TableCredentialsVariants::Gcp)
        } else {
            None
        }
    }
}

pub enum TableCredentialsVariants {
    Aws(TableCredentialsAws),
    Azure(TableCredentialsAzure),
    Gcp(TableCredentialsGcp),
}

#[derive(Debug, serde::Deserialize)]
pub struct TableCredentialsAws {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,

    #[serde(default)]
    pub access_point: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
pub struct TableCredentialsAzure {
    pub sas_token: String,
}

#[derive(Debug, serde::Deserialize)]
pub struct TableCredentialsGcp {
    pub oauth_token: String,
}

fn null_to_default<'de, T, D>(d: D) -> Result<T, D::Error>
where
    T: Default + serde::de::Deserialize<'de>,
    D: serde::de::Deserializer<'de>,
{
    use serde::Deserialize;
    let opt_val = Option::<T>::deserialize(d)?;
    Ok(opt_val.unwrap_or_default())
}
