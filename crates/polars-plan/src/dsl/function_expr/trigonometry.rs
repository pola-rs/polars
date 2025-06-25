use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum TrigonometricFunction {
    Cos,
    Cot,
    Sin,
    Tan,
    ArcCos,
    ArcSin,
    ArcTan,
    Cosh,
    Sinh,
    Tanh,
    ArcCosh,
    ArcSinh,
    ArcTanh,
    Degrees,
    Radians,
}

impl Display for TrigonometricFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use self::*;
        match self {
            TrigonometricFunction::Cos => write!(f, "cos"),
            TrigonometricFunction::Cot => write!(f, "cot"),
            TrigonometricFunction::Sin => write!(f, "sin"),
            TrigonometricFunction::Tan => write!(f, "tan"),
            TrigonometricFunction::ArcCos => write!(f, "arccos"),
            TrigonometricFunction::ArcSin => write!(f, "arcsin"),
            TrigonometricFunction::ArcTan => write!(f, "arctan"),
            TrigonometricFunction::Cosh => write!(f, "cosh"),
            TrigonometricFunction::Sinh => write!(f, "sinh"),
            TrigonometricFunction::Tanh => write!(f, "tanh"),
            TrigonometricFunction::ArcCosh => write!(f, "arccosh"),
            TrigonometricFunction::ArcSinh => write!(f, "arcsinh"),
            TrigonometricFunction::ArcTanh => write!(f, "arctanh"),
            TrigonometricFunction::Degrees => write!(f, "degrees"),
            TrigonometricFunction::Radians => write!(f, "radians"),
        }
    }
}

impl From<TrigonometricFunction> for FunctionExpr {
    fn from(value: TrigonometricFunction) -> Self {
        Self::Trigonometry(value)
    }
}
