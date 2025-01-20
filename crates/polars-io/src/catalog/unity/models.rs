#[derive(Debug, serde::Deserialize)]
pub struct CatalogInfo {
    pub name: String,
    pub comment: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
pub struct SchemaInfo {
    pub name: String,
    pub comment: Option<String>,
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
}

#[derive(Debug, strum_macros::Display, serde::Deserialize)]
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

#[derive(Debug, strum_macros::Display, serde::Deserialize)]
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

#[derive(Debug, serde::Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub type_text: String,
    pub type_interval_type: Option<String>,
    pub position: Option<u32>,
    pub comment: Option<String>,
    pub partition_index: Option<u32>,
}
