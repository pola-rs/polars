use std::ops::{Add, BitAnd, BitXor, Sub};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[derive(Clone, PartialEq, Hash, Debug, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Selector {
    Add(Box<Selector>, Box<Selector>),
    Sub(Box<Selector>, Box<Selector>),
    ExclusiveOr(Box<Selector>, Box<Selector>),
    InterSect(Box<Selector>, Box<Selector>),
    Root(Box<Expr>),
}

impl Selector {
    pub fn new(e: Expr) -> Self {
        Self::Root(Box::new(e))
    }
}

impl Add for Selector {
    type Output = Selector;

    fn add(self, rhs: Self) -> Self::Output {
        Selector::Add(Box::new(self), Box::new(rhs))
    }
}

impl BitAnd for Selector {
    type Output = Selector;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Selector::InterSect(Box::new(self), Box::new(rhs))
    }
}

impl BitXor for Selector {
    type Output = Selector;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Selector::ExclusiveOr(Box::new(self), Box::new(rhs))
    }
}

impl Sub for Selector {
    type Output = Selector;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self::Output {
        Selector::Sub(Box::new(self), Box::new(rhs))
    }
}

impl From<&str> for Selector {
    fn from(value: &str) -> Self {
        Selector::new(col(value))
    }
}

impl From<String> for Selector {
    fn from(value: String) -> Self {
        Selector::new(col(value.as_ref()))
    }
}

impl From<ColumnName> for Selector {
    fn from(value: ColumnName) -> Self {
        Selector::new(Expr::Column(value))
    }
}

impl From<Expr> for Selector {
    fn from(value: Expr) -> Self {
        Selector::new(value)
    }
}
