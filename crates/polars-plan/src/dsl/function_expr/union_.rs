use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug, Hash)]
pub enum UnionFunction {
    Tag,
    Field(PlSmallStr),
    /// Every row holds the input as the variant `variant` of the union `dtype`.
    Value {
        dtype: DataTypeExpr,
        variant: PlSmallStr,
    },
}

impl Display for UnionFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            UnionFunction::Tag => write!(f, "union.tag"),
            UnionFunction::Field(name) => write!(f, "union.field({name})"),
            UnionFunction::Value { variant, .. } => write!(f, "union.value({variant})"),
        }
    }
}

impl From<UnionFunction> for FunctionExpr {
    fn from(value: UnionFunction) -> Self {
        Self::UnionExpr(value)
    }
}
