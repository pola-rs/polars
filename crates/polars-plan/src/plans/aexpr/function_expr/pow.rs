use std::fmt;

use super::IRFunctionExpr;
use crate::prelude::FunctionOptions;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum IRPowFunction {
    Generic,
    Sqrt,
    Cbrt,
}

impl IRPowFunction {
    pub fn function_options(&self) -> FunctionOptions {
        use IRPowFunction as P;
        match self {
            P::Generic | P::Sqrt | P::Cbrt => FunctionOptions::elementwise(),
        }
    }
}

impl fmt::Display for IRPowFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        use self::*;
        match self {
            IRPowFunction::Generic => write!(f, "pow"),
            IRPowFunction::Sqrt => write!(f, "sqrt"),
            IRPowFunction::Cbrt => write!(f, "cbrt"),
        }
    }
}

impl From<IRPowFunction> for IRFunctionExpr {
    fn from(value: IRPowFunction) -> Self {
        Self::Pow(value)
    }
}
