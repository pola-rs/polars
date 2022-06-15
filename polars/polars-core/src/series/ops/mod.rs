#[cfg(feature = "diff")]
#[cfg_attr(docsrs, doc(cfg(feature = "diff")))]
pub mod diff;
mod downcast;
#[cfg(feature = "ewma")]
#[cfg_attr(docsrs, doc(cfg(feature = "ewma")))]
mod ewm;
mod extend;
#[cfg(feature = "moment")]
#[cfg_attr(docsrs, doc(cfg(feature = "moment")))]
pub mod moment;
mod null;
#[cfg(feature = "pct_change")]
#[cfg_attr(docsrs, doc(cfg(feature = "pct_change")))]
pub mod pct_change;
#[cfg(feature = "round_series")]
#[cfg_attr(docsrs, doc(cfg(feature = "round_series")))]
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

impl Default for NullBehavior {
    fn default() -> Self {
        NullBehavior::Ignore
    }
}
