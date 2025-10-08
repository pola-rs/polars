use std::fmt;
use std::hash::Hash;
use std::num::NonZeroU64;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[repr(transparent)]
pub struct Dimension(NonZeroU64);

/// A dimension in a reshape.
///
/// Any dimension smaller than 0 is seen as an `infer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ReshapeDimension {
    Infer,
    Specified(Dimension),
}

impl fmt::Debug for Dimension {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(f)
    }
}

impl fmt::Display for ReshapeDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Infer => f.write_str("inferred"),
            Self::Specified(v) => v.get().fmt(f),
        }
    }
}

impl Hash for ReshapeDimension {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_repr().hash(state)
    }
}

impl Dimension {
    #[inline]
    pub const fn new(v: u64) -> Self {
        assert!(v <= i64::MAX as u64);

        // SAFETY: Bounds check done before
        let dim = unsafe { NonZeroU64::new_unchecked(v.wrapping_add(1)) };
        Self(dim)
    }

    #[inline]
    pub const fn get(self) -> u64 {
        self.0.get() - 1
    }
}

impl ReshapeDimension {
    #[inline]
    pub const fn new(v: i64) -> Self {
        if v < 0 {
            Self::Infer
        } else {
            // SAFETY: We have bounds checked for -1
            let dim = unsafe { NonZeroU64::new_unchecked((v as u64).wrapping_add(1)) };
            Self::Specified(Dimension(dim))
        }
    }

    #[inline]
    fn to_repr(self) -> u64 {
        match self {
            Self::Infer => 0,
            Self::Specified(dim) => dim.0.get(),
        }
    }

    #[inline]
    pub const fn get(self) -> Option<u64> {
        match self {
            ReshapeDimension::Infer => None,
            ReshapeDimension::Specified(dim) => Some(dim.get()),
        }
    }

    #[inline]
    pub const fn get_or_infer(self, inferred: u64) -> u64 {
        match self {
            ReshapeDimension::Infer => inferred,
            ReshapeDimension::Specified(dim) => dim.get(),
        }
    }

    #[inline]
    pub fn get_or_infer_with(self, f: impl Fn() -> u64) -> u64 {
        match self {
            ReshapeDimension::Infer => f(),
            ReshapeDimension::Specified(dim) => dim.get(),
        }
    }

    pub const fn new_dimension(dimension: u64) -> ReshapeDimension {
        Self::Specified(Dimension::new(dimension))
    }
}

impl TryFrom<i64> for Dimension {
    type Error = ();

    #[inline]
    fn try_from(value: i64) -> Result<Self, Self::Error> {
        let ReshapeDimension::Specified(v) = ReshapeDimension::new(value) else {
            return Err(());
        };

        Ok(v)
    }
}
