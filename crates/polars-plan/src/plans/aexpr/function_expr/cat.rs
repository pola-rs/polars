use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum IRCategoricalFunction {
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
    To(DataType, bool),
    Physical,
}

impl<'a> FieldsMapper<'a> {
    /// Validate that the dtype is Categorical or Enum.
    pub fn ensure_is_categorical(&self) -> PolarsResult<()> {
        let dt = self.args()[0].dtype();
        polars_ensure!(
            dt.is_categorical() || dt.is_enum() || matches!(dt, DataType::Unknown(UnknownKind::Any)),
            InvalidOperation:
            "expected Categorical or Enum data type for categorical operation, got: {dt:?}"
        );
        Ok(())
    }
}

impl IRCategoricalFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use IRCategoricalFunction::*;
        // `cat.to` builds a categorical from its integer physical representation.
        if !matches!(self, To(..)) {
            mapper.ensure_is_categorical()?;
        }
        match self {
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
            To(dt, _strict) => mapper.with_dtype(dt.clone()),
            Physical => mapper.try_map_dtype(|dt| Ok(DataType::from(dt.cat_physical()?))),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        FunctionOptions::elementwise()
    }
}

impl Display for IRCategoricalFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRCategoricalFunction::*;
        let s = match self {
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
            To(_, _) => "to",
            Physical => "physical",
        };
        write!(f, "cat.{s}")
    }
}

impl From<IRCategoricalFunction> for IRFunctionExpr {
    fn from(func: IRCategoricalFunction) -> Self {
        IRFunctionExpr::Categorical(func)
    }
}
