#[cfg(feature = "diff")]
pub mod diff;
mod downcast;
#[cfg(feature = "ewma")]
mod ewm;
mod extend;
#[cfg(feature = "moment")]
pub mod moment;
mod null;
#[cfg(feature = "pct_change")]
pub mod pct_change;
#[cfg(feature = "round_series")]
mod round;
mod to_list;
mod unique;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NullBehavior {
    /// drop nulls
    Drop,
    /// ignore nulls
    #[default]
    Ignore,
}
