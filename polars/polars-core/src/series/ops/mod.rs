#[cfg(feature = "diff")]
pub mod diff;
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

#[derive(Copy, Clone)]
pub enum NullBehavior {
    /// drop nulls
    Drop,
    /// ignore nulls
    Ignore,
}
