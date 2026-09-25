use super::*;

/// A SQL operation whose semantics depend on the operand dtypes. Only the SQL frontend
/// creates these; they are lowered to ordinary expressions during DSL→IR conversion,
/// once the dtypes are known, so every SQL entry point gets the same result.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum SqlFunction {
    /// `lhs <op> rhs`. A decimal literal (or a `CASE` of them) combined with a float
    /// takes the float's type, as a float literal would.
    Binary(SqlBinaryOp),
    /// `IN`: a list of decimal literals tested against a float takes the float's type.
    #[cfg(feature = "is_in")]
    IsIn { nulls_equal: bool },
    /// A decimal as `Float64`, for SQL functions without decimal implementations.
    ToFloat,
    /// `ROUND`: decimals round half away from zero, as in Postgres.
    #[cfg(feature = "round_series")]
    Round { decimals: u32 },
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum SqlBinaryOp {
    Plus,
    Minus,
    Mul,
    Div,
    /// The remainder of truncating division, with the dividend's sign.
    Rem,
    /// The integer quotient rounded toward zero.
    IntDiv,
    Eq,
    EqMissing,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

impl Display for SqlFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use SqlBinaryOp as B;
        let name = match self {
            Self::Binary(op) => match op {
                B::Plus => "sql_add",
                B::Minus => "sql_sub",
                B::Mul => "sql_mul",
                B::Div => "sql_div",
                B::Rem => "sql_rem",
                B::IntDiv => "sql_int_div",
                B::Eq => "sql_eq",
                B::EqMissing => "sql_eq_missing",
                B::Lt => "sql_lt",
                B::LtEq => "sql_lt_eq",
                B::Gt => "sql_gt",
                B::GtEq => "sql_gt_eq",
            },
            #[cfg(feature = "is_in")]
            Self::IsIn { .. } => "sql_is_in",
            Self::ToFloat => "sql_to_float",
            #[cfg(feature = "round_series")]
            Self::Round { .. } => "sql_round",
        };
        f.write_str(name)
    }
}
