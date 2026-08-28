use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum IRUuidFunction {
    GenerateV4,
    GenerateV7,
    Version,
    #[cfg(feature = "dtype-datetime")]
    Timestamp {
        strict: bool,
    },
}

impl IRUuidFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        match self {
            Self::GenerateV4 | Self::GenerateV7 => mapper.with_dtype(DataType::Uuid),
            Self::Version => mapper.with_dtype(DataType::UInt8),
            #[cfg(feature = "dtype-datetime")]
            Self::Timestamp { .. } => mapper.with_dtype(DataType::Datetime(
                TimeUnit::Milliseconds,
                Some(TimeZone::UTC),
            )),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        FunctionOptions::elementwise()
    }
}

impl Display for IRUuidFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::GenerateV4 => "generate_v4",
            Self::GenerateV7 => "generate_v7",
            Self::Version => "version",
            #[cfg(feature = "dtype-datetime")]
            Self::Timestamp { .. } => "timestamp",
        };
        write!(f, "uuid.{name}")
    }
}

impl From<IRUuidFunction> for IRFunctionExpr {
    fn from(value: IRUuidFunction) -> Self {
        IRFunctionExpr::UuidExpr(value)
    }
}
