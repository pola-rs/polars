use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub enum IRMapFunction {
    Entries,
    Keys,
    Values,
    Length,
    ContainsKey,
    Get,
}

impl<'a> FieldsMapper<'a> {
    /// Map the key and value dtypes of a Map, keeping the Map column's name.
    ///
    /// Both the unary and the binary map functions take the Map as their first argument.
    fn try_map_map_dtype(
        self,
        op: &str,
        func: impl FnOnce(&DataType, &DataType) -> PolarsResult<DataType>,
    ) -> PolarsResult<Field> {
        self.try_map_dtype(|dtype| {
            let Some((key, value)) = dtype.as_map() else {
                polars_bail!(InvalidOperation: "`{op}` requires a Map dtype, got `{dtype}`");
            };
            func(key, value)
        })
    }
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
            Keys => mapper.try_map_map_dtype("map.keys", |key, _| {
                Ok(DataType::List(Box::new(key.clone())))
            }),
            Values => mapper.try_map_map_dtype("map.values", |_, value| {
                Ok(DataType::List(Box::new(value.clone())))
            }),
            Length => mapper.try_map_map_dtype("map.len", |_, _| Ok(IDX_DTYPE)),
            ContainsKey => {
                mapper.try_map_map_dtype("map.contains_key", |_, _| Ok(DataType::Boolean))
            },
            Get => mapper.try_map_map_dtype("map.get", |_, value| Ok(value.clone())),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        use IRMapFunction::*;
        match self {
            Entries | Keys | Values | Length | ContainsKey | Get => FunctionOptions::elementwise(),
        }
    }
}

impl Display for IRMapFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRMapFunction::*;

        let name = match self {
            Entries => "entries",
            Keys => "keys",
            Values => "values",
            Length => "len",
            ContainsKey => "contains_key",
            Get => "get",
        };
        write!(f, "map.{name}")
    }
}

impl From<IRMapFunction> for IRFunctionExpr {
    fn from(value: IRMapFunction) -> Self {
        Self::MapExpr(value)
    }
}
