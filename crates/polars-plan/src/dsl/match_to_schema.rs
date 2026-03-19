use super::{Expr, ExtraColumnsPolicy, MissingColumnsPolicy};

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum MissingColumnsPolicyOrExpr {
    Insert,
    Raise,
    InsertWith(Expr),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum UpcastOrForbid {
    Upcast,
    Forbid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct MatchToSchemaPerColumn {
    pub missing_columns: MissingColumnsPolicyOrExpr,
    pub missing_struct_fields: MissingColumnsPolicy,

    pub extra_struct_fields: ExtraColumnsPolicy,

    pub integer_cast: UpcastOrForbid,
    pub float_cast: UpcastOrForbid,
}
