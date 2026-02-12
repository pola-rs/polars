use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum IRExtensionFunction {
    To(DataType),
    Storage,
}

impl IRExtensionFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use IRExtensionFunction::*;
        match self {
            To(dtype) => mapper.with_dtype(dtype.clone()),
            Storage => mapper.map_dtype(|dt| match dt {
                DataType::Extension(_, storage) => (**storage).clone(),
                dt => dt.clone(),
            }),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        use IRExtensionFunction::*;
        match self {
            To(_dtype) => FunctionOptions::elementwise(),
            Storage => FunctionOptions::elementwise(),
        }
    }
}

impl Display for IRExtensionFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRExtensionFunction::*;
        match self {
            To(dtype) => write!(f, "ext.to({dtype:?})"),
            Storage => write!(f, "ext.storage()"),
        }
    }
}

impl From<IRExtensionFunction> for IRFunctionExpr {
    fn from(func: IRExtensionFunction) -> Self {
        IRFunctionExpr::Extension(func)
    }
}
