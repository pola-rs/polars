use std::ops::{Add, BitAnd, Sub};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[derive(Clone, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Selector {
    Add(Box<Selector>, Box<Selector>),
    Sub(Box<Selector>, Box<Selector>),
    InterSect(Box<Selector>, Box<Selector>),
    Root(Box<Expr>),
}

impl Selector {
    pub(crate) fn new(e: Expr) -> Self {
        Self::Root(Box::new(e))
    }
}

impl Add for Selector {
    type Output = Selector;

    fn add(self, rhs: Self) -> Self::Output {
        Selector::Add(Box::new(self), Box::new(rhs))
    }
}

impl Sub for Selector {
    type Output = Selector;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self::Output {
        Selector::Sub(Box::new(self), Box::new(rhs))
    }
}

impl BitAnd for Selector {
    type Output = Selector;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Selector::InterSect(Box::new(self), Box::new(rhs))
    }
}
