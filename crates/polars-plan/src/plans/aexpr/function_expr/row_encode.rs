#[derive(Clone, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum RowEncodingVariant {
    Unordered,
    Ordered {
        descending: Option<Vec<bool>>,
        nulls_last: Option<Vec<bool>>,
        broadcast_nulls: Option<bool>,
    },
}
