use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum IRTrigonometricFunction {
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

impl Display for IRTrigonometricFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use self::*;
        match self {
            IRTrigonometricFunction::Cos => write!(f, "cos"),
            IRTrigonometricFunction::Cot => write!(f, "cot"),
            IRTrigonometricFunction::Sin => write!(f, "sin"),
            IRTrigonometricFunction::Tan => write!(f, "tan"),
            IRTrigonometricFunction::ArcCos => write!(f, "arccos"),
            IRTrigonometricFunction::ArcSin => write!(f, "arcsin"),
            IRTrigonometricFunction::ArcTan => write!(f, "arctan"),
            IRTrigonometricFunction::Cosh => write!(f, "cosh"),
            IRTrigonometricFunction::Sinh => write!(f, "sinh"),
            IRTrigonometricFunction::Tanh => write!(f, "tanh"),
            IRTrigonometricFunction::ArcCosh => write!(f, "arccosh"),
            IRTrigonometricFunction::ArcSinh => write!(f, "arcsinh"),
            IRTrigonometricFunction::ArcTanh => write!(f, "arctanh"),
            IRTrigonometricFunction::Degrees => write!(f, "degrees"),
            IRTrigonometricFunction::Radians => write!(f, "radians"),
        }
    }
}

impl From<IRTrigonometricFunction> for IRFunctionExpr {
    fn from(value: IRTrigonometricFunction) -> Self {
        Self::Trigonometry(value)
    }
}
