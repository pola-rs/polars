#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug, Hash)]
pub enum BinaryFunction {
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
    /// The parameters are destination type, and whether to use little endian
    /// encoding.
    Reinterpret(DataTypeExpr, bool),
    Slice,
    Head,
    Tail,
}

impl Display for BinaryFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use BinaryFunction::*;
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
        };
        write!(f, "bin.{s}")
    }
}
