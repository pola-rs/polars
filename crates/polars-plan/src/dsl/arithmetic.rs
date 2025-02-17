use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

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

impl Neg for Expr {
    type Output = Expr;

    fn neg(self) -> Self::Output {
        self.map_private(FunctionExpr::Negate)
    }
}

impl Expr {
    /// Floor divide `self` by `rhs`.
    pub fn floor_div(self, rhs: Self) -> Self {
        binary_expr(self, Operator::FloorDivide, rhs)
    }

    /// Raise expression to the power `exponent`
    pub fn pow<E: Into<Expr>>(self, exponent: E) -> Self {
        self.map_many_private(
            FunctionExpr::Pow(PowFunction::Generic),
            &[exponent.into()],
            false,
            None,
        )
    }

    /// Compute the square root of the given expression
    pub fn sqrt(self) -> Self {
        self.map_private(FunctionExpr::Pow(PowFunction::Sqrt))
    }

    /// Compute the cube root of the given expression
    pub fn cbrt(self) -> Self {
        self.map_private(FunctionExpr::Pow(PowFunction::Cbrt))
    }

    /// Compute the cosine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn cos(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::Cos))
    }

    /// Compute the cotangent of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn cot(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::Cot))
    }

    /// Compute the sine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn sin(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::Sin))
    }

    /// Compute the tangent of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn tan(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::Tan))
    }

    /// Compute the inverse cosine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arccos(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::ArcCos))
    }

    /// Compute the inverse sine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arcsin(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::ArcSin))
    }

    /// Compute the inverse tangent of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arctan(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::ArcTan))
    }

    /// Compute the inverse tangent of the given expression, with the angle expressed as the argument of a complex number
    #[cfg(feature = "trigonometry")]
    pub fn arctan2(self, x: Self) -> Self {
        self.map_many_private(FunctionExpr::Atan2, &[x], false, None)
    }

    /// Compute the hyperbolic cosine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn cosh(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::Cosh))
    }

    /// Compute the hyperbolic sine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn sinh(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::Sinh))
    }

    /// Compute the hyperbolic tangent of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn tanh(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::Tanh))
    }

    /// Compute the inverse hyperbolic cosine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arccosh(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::ArcCosh))
    }

    /// Compute the inverse hyperbolic sine of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arcsinh(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::ArcSinh))
    }

    /// Compute the inverse hyperbolic tangent of the given expression
    #[cfg(feature = "trigonometry")]
    pub fn arctanh(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::ArcTanh))
    }

    /// Convert from radians to degrees
    #[cfg(feature = "trigonometry")]
    pub fn degrees(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::Degrees))
    }

    /// Convert from degrees to radians
    #[cfg(feature = "trigonometry")]
    pub fn radians(self) -> Self {
        self.map_private(FunctionExpr::Trigonometry(TrigonometricFunction::Radians))
    }

    /// Compute the sign of the given expression
    #[cfg(feature = "sign")]
    pub fn sign(self) -> Self {
        self.map_private(FunctionExpr::Sign)
    }
}
