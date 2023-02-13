use std::ops::{Add, Div, Mul, Rem, Sub};

use super::*;

// Arithmetic ops
impl Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Self) -> Self::Output {
        binary_expr(self, Operator::Plus, rhs)
    }
}

impl Sub for Expr {
    type Output = Expr;

    fn sub(self, rhs: Self) -> Self::Output {
        binary_expr(self, Operator::Minus, rhs)
    }
}

impl Div for Expr {
    type Output = Expr;

    fn div(self, rhs: Self) -> Self::Output {
        binary_expr(self, Operator::Divide, rhs)
    }
}

impl Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Self) -> Self::Output {
        binary_expr(self, Operator::Multiply, rhs)
    }
}

impl Rem for Expr {
    type Output = Expr;

    fn rem(self, rhs: Self) -> Self::Output {
        binary_expr(self, Operator::Modulus, rhs)
    }
}

impl Expr {
    /// Floor divide `self` by `rhs`.
    pub fn floor_div(self, rhs: Self) -> Self {
        binary_expr(self, Operator::FloorDivide, rhs)
    }

    /// Raise expression to the power `exponent`
    pub fn pow<E: Into<Expr>>(self, exponent: E) -> Self {
        Expr::Function {
            input: vec![self, exponent.into()],
            function: FunctionExpr::Pow,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the sine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn sin(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::Sin),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the cosine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn cos(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::Cos),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the tangent of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn tan(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::Tan),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the inverse sine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arcsin(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::ArcSin),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the inverse cosine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arccos(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::ArcCos),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the inverse tangent of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arctan(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::ArcTan),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the hyperbolic sine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn sinh(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::Sinh),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the hyperbolic cosine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn cosh(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::Cosh),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the hyperbolic tangent of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn tanh(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::Tanh),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the inverse hyperbolic sine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arcsinh(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::ArcSinh),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the inverse hyperbolic cosine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arccosh(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::ArcCosh),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }

    /// Compute the inverse hyperbolic tangent of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arctanh(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Trigonometry(TrigonometricFunction::ArcTanh),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                fmt_str: "arctanh",
                ..Default::default()
            },
        }
    }

    /// Compute the sign of the given expression
    #[cfg(feature = "sign")]
    pub fn sign(self) -> Self {
        Expr::Function {
            input: vec![self],
            function: FunctionExpr::Sign,
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                ..Default::default()
            },
        }
    }
}
