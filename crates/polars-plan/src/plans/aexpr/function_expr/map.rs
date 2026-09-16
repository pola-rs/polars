use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub enum IRMapFunction {
    Entries,
}

impl IRMapFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use IRMapFunction::*;
        match self {
            Entries => mapper.try_map_dtype(|dtype| {
                let Some(entries) = dtype.map_storage_dtype() else {
                    polars_bail!(
                        InvalidOperation:
                        "`map.entries` requires a Map dtype, got `{dtype}`",
                    );
                };
                Ok(entries)
            }),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        use IRMapFunction::*;
        match self {
            Entries => FunctionOptions::elementwise(),
        }
    }
}

impl Display for IRMapFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRMapFunction::*;

        let name = match self {
            Entries => "entries",
        };
        write!(f, "map.{name}")
    }
}

impl From<IRMapFunction> for IRFunctionExpr {
    fn from(value: IRMapFunction) -> Self {
        Self::MapExpr(value)
    }
}
