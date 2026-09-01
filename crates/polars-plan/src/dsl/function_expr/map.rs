use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub enum MapFunction {
    Entries,
}

impl Display for MapFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use MapFunction::*;

        let name = match self {
            Entries => "entries",
        };
        write!(f, "map.{name}")
    }
}

impl From<MapFunction> for FunctionExpr {
    fn from(value: MapFunction) -> Self {
        Self::MapExpr(value)
    }
}
