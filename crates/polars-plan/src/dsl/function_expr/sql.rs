use super::*;

/// A SQL operation whose semantics depend on the operand dtypes. Only the SQL frontend
/// creates these; they are lowered to ordinary expressions during DSL→IR conversion,
/// once the dtypes are known, so every SQL entry point gets the same result.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum SqlFunction {
    /// `lhs <op> rhs`.
    Binary(SqlBinaryOp),
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum SqlBinaryOp {
    Mul,
    Div,
}

impl Display for SqlFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Binary(SqlBinaryOp::Mul) => "sql_mul",
            Self::Binary(SqlBinaryOp::Div) => "sql_div",
        };
        f.write_str(name)
    }
}
