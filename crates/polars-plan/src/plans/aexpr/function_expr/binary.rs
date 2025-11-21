use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum IRBinaryFunction {
    Contains,
    StartsWith,
    EndsWith,
    #[cfg(feature = "binary_encoding")]
    HexDecode(bool),
    #[cfg(feature = "binary_encoding")]
    HexEncode,
    #[cfg(feature = "binary_encoding")]
    Base64Decode(bool),
    #[cfg(feature = "binary_encoding")]
    Base64Encode,
    Size,
    #[cfg(feature = "binary_encoding")]
    Reinterpret(DataType, bool),
}

impl IRBinaryFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use IRBinaryFunction::*;
        match self {
            Contains => mapper.with_dtype(DataType::Boolean),
            EndsWith | StartsWith => mapper.with_dtype(DataType::Boolean),
            #[cfg(feature = "binary_encoding")]
            HexDecode(_) | Base64Decode(_) => mapper.with_same_dtype(),
            #[cfg(feature = "binary_encoding")]
            HexEncode | Base64Encode => mapper.with_dtype(DataType::String),
            Size => mapper.with_dtype(DataType::UInt32),
            #[cfg(feature = "binary_encoding")]
            Reinterpret(dtype, _) => mapper.with_dtype(dtype.clone()),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        use IRBinaryFunction as B;
        match self {
            B::Contains | B::StartsWith | B::EndsWith => {
                FunctionOptions::elementwise().with_supertyping(Default::default())
            },
            B::Size => FunctionOptions::elementwise(),
            #[cfg(feature = "binary_encoding")]
            B::HexDecode(_)
            | B::HexEncode
            | B::Base64Decode(_)
            | B::Base64Encode
            | B::Reinterpret(_, _) => FunctionOptions::elementwise(),
        }
    }
}

impl Display for IRBinaryFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRBinaryFunction::*;
        let s = match self {
            Contains => "contains",
            StartsWith => "starts_with",
            EndsWith => "ends_with",
            #[cfg(feature = "binary_encoding")]
            HexDecode(_) => "hex_decode",
            #[cfg(feature = "binary_encoding")]
            HexEncode => "hex_encode",
            #[cfg(feature = "binary_encoding")]
            Base64Decode(_) => "base64_decode",
            #[cfg(feature = "binary_encoding")]
            Base64Encode => "base64_encode",
            Size => "size_bytes",
            #[cfg(feature = "binary_encoding")]
            Reinterpret(_, _) => "reinterpret",
        };
        write!(f, "bin.{s}")
    }
}

impl From<IRBinaryFunction> for IRFunctionExpr {
    fn from(b: IRBinaryFunction) -> Self {
        IRFunctionExpr::BinaryExpr(b)
    }
}
