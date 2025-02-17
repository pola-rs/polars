mod downcast;
mod extend;
mod null;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
mod reshape;

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NullBehavior {
    /// drop nulls
    Drop,
    /// ignore nulls
    #[default]
    Ignore,
}
