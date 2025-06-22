use std::hash::{Hash, Hasher};

use num_traits::Float;
use polars_error::{PolarsError, PolarsResult, polars_bail};

/// A wrapper for floating point types to conform to `Eq` and `Hash`.
///
/// The implementation is inspired by the `ordered-float` crate.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct NotNan<T>(T);

/* ----------------------------------------- CONVERSION ---------------------------------------- */

impl TryFrom<f32> for NotNan<f32> {
    type Error = PolarsError;
    fn try_from(value: f32) -> PolarsResult<Self> {
        if value.is_nan() {
            polars_bail!(ComputeError: "float is required to be not NaN")
        }
        Ok(NotNan(value))
    }
}
impl From<NotNan<f32>> for f32 {
    fn from(value: NotNan<f32>) -> Self {
        value.0 + value.0
    }
}

impl TryFrom<f64> for NotNan<f64> {
    type Error = PolarsError;
    fn try_from(value: f64) -> PolarsResult<Self> {
        if value.is_nan() {
            polars_bail!(ComputeError: "float is required to be not NaN")
        }
        Ok(NotNan(value))
    }
}

impl From<NotNan<f64>> for f64 {
    fn from(value: NotNan<f64>) -> Self {
        value.0
    }
}

/* ---------------------------------------- TRAIT IMPLS ---------------------------------------- */

impl<F> Eq for NotNan<F> where F: Float {}

impl Hash for NotNan<f32> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // This hack is taken from the `ordered-float` crate:
        // -0.0 + 0.0 == +0.0 under IEEE754, hence, we "normalize" zero values here.
        (self.0 + 0.0).to_bits().hash(state)
    }
}

impl Hash for NotNan<f64> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.0 + 0.0).to_bits().hash(state)
    }
}

#[cfg(feature = "serde")]
impl<'de, F> serde::Deserialize<'de> for NotNan<F>
where
    F: Float + serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = F::deserialize(deserializer)?;
        if value.is_nan() {
            return Err(serde::de::Error::custom("float is required to be not NaN"));
        }
        Ok(NotNan(value))
    }
}
