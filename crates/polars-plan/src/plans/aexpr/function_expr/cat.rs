use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum IRCategoricalFunction {
    GetCategories,
    #[cfg(feature = "strings")]
    LenBytes,
    #[cfg(feature = "strings")]
    LenChars,
    #[cfg(feature = "strings")]
    StartsWith(String),
    #[cfg(feature = "strings")]
    EndsWith(String),
    #[cfg(feature = "strings")]
    Slice(i64, Option<usize>),
}

impl IRCategoricalFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use IRCategoricalFunction::*;
        match self {
            GetCategories => mapper.with_dtype(DataType::String),
            #[cfg(feature = "strings")]
            LenBytes => mapper.with_dtype(DataType::UInt32),
            #[cfg(feature = "strings")]
            LenChars => mapper.with_dtype(DataType::UInt32),
            #[cfg(feature = "strings")]
            StartsWith(_) => mapper.with_dtype(DataType::Boolean),
            #[cfg(feature = "strings")]
            EndsWith(_) => mapper.with_dtype(DataType::Boolean),
            #[cfg(feature = "strings")]
            Slice(_, _) => mapper.with_dtype(DataType::String),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        use IRCategoricalFunction as C;
        match self {
            C::GetCategories => FunctionOptions::groupwise(),
            #[cfg(feature = "strings")]
            C::LenBytes | C::LenChars | C::StartsWith(_) | C::EndsWith(_) | C::Slice(_, _) => {
                FunctionOptions::elementwise()
            },
        }
    }
}

impl Display for IRCategoricalFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRCategoricalFunction::*;
        let s = match self {
            GetCategories => "get_categories",
            #[cfg(feature = "strings")]
            LenBytes => "len_bytes",
            #[cfg(feature = "strings")]
            LenChars => "len_chars",
            #[cfg(feature = "strings")]
            StartsWith(_) => "starts_with",
            #[cfg(feature = "strings")]
            EndsWith(_) => "ends_with",
            #[cfg(feature = "strings")]
            Slice(_, _) => "slice",
        };
        write!(f, "cat.{s}")
    }
}

impl From<IRCategoricalFunction> for IRFunctionExpr {
    fn from(func: IRCategoricalFunction) -> Self {
        IRFunctionExpr::Categorical(func)
    }
}
