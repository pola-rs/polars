use std::ops::{Add, Sub};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Selector {
    pub(crate) add: Vec<Expr>,
    pub(crate) subtract: Vec<Expr>,
}

impl Selector {
    pub(crate) fn new(e: Expr) -> Self {
        Self {
            add: vec![e],
            subtract: vec![],
        }
    }
}

impl Add for &Selector {
    type Output = Selector;

    fn add(self, rhs: Self) -> Self::Output {
        let mut add = Vec::with_capacity(self.add.len() + rhs.add.len());
        add.extend_from_slice(&self.add);
        add.extend_from_slice(&rhs.add);

        let mut subtract = Vec::with_capacity(self.subtract.len() + rhs.subtract.len());
        subtract.extend_from_slice(&self.subtract);
        subtract.extend_from_slice(&rhs.subtract);
        Selector { add, subtract }
    }
}

impl Sub for &Selector {
    type Output = Selector;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut subtract = Vec::with_capacity(self.subtract.len() + rhs.subtract.len());
        subtract.extend_from_slice(&self.subtract);
        subtract.extend_from_slice(&rhs.subtract);
        Selector {
            add: self.add.clone(),
            subtract,
        }
    }
}
