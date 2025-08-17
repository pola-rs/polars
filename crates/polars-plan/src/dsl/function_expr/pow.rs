use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum PowFunction {
    Generic,
    Sqrt,
    Cbrt,
}

impl Display for PowFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use self::*;
        match self {
            PowFunction::Generic => write!(f, "pow"),
            PowFunction::Sqrt => write!(f, "sqrt"),
            PowFunction::Cbrt => write!(f, "cbrt"),
        }
    }
}

impl From<PowFunction> for FunctionExpr {
    fn from(value: PowFunction) -> Self {
        Self::Pow(value)
    }
}
