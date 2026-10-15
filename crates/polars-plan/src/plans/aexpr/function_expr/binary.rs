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
    Slice,
    Head,
    Tail,
    Get(bool),
}

impl<'a> FieldsMapper<'a> {
    /// Validate that the dtype is Binary.
    pub fn ensure_is_binary(self) -> PolarsResult<Self> {
        let dt = self.args()[0].dtype();
        polars_ensure!(
            matches!(dt, DataType::Binary | DataType::Unknown(UnknownKind::Any)),
            InvalidOperation:
            "expected Binary data type for binary operation, got: {dt:?}"
        );
        Ok(self)
    }
}

impl IRBinaryFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use IRBinaryFunction::*;
        match self {
            Contains => mapper.ensure_is_binary()?.with_dtype(DataType::Boolean),
            EndsWith | StartsWith => mapper.ensure_is_binary()?.with_dtype(DataType::Boolean),
            #[cfg(feature = "binary_encoding")]
            HexDecode(_) | Base64Decode(_) => mapper.ensure_is_binary()?.with_same_dtype(),
            #[cfg(feature = "binary_encoding")]
            HexEncode | Base64Encode => mapper.ensure_is_binary()?.with_dtype(DataType::String),
            Size => mapper.ensure_is_binary()?.with_dtype(DataType::UInt32),
            #[cfg(feature = "binary_encoding")]
            Reinterpret(dtype, _) => mapper.ensure_is_binary()?.with_dtype(dtype.clone()),
            Slice | Head | Tail => mapper.ensure_is_binary()?.with_same_dtype(),
            Get(_) => mapper.ensure_is_binary()?.with_dtype(DataType::UInt8),
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
            B::Slice | B::Head | B::Tail | B::Get(_) => FunctionOptions::elementwise(),
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
            Slice => "slice",
            Head => "head",
            Tail => "tail",
            Get(_) => "get",
        };
        write!(f, "bin.{s}")
    }
}

impl From<IRBinaryFunction> for IRFunctionExpr {
    fn from(b: IRBinaryFunction) -> Self {
        IRFunctionExpr::BinaryExpr(b)
    }
}
