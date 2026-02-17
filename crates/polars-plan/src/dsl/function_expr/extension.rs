use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug, Hash)]
pub enum ExtensionFunction {
    To(DataTypeExpr),
    Storage,
}

impl Display for ExtensionFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use ExtensionFunction::*;
        match self {
            To(dtype) => write!(f, "ext.to({dtype:?})"),
            Storage => write!(f, "ext.storage()"),
        }
    }
}

impl From<ExtensionFunction> for FunctionExpr {
    fn from(func: ExtensionFunction) -> Self {
        FunctionExpr::Extension(func)
    }
}
