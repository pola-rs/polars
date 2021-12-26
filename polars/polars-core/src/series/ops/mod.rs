#[cfg(feature = "diff")]
pub mod diff;
#[cfg(feature = "ewma")]
mod ewm;
mod extend;
#[cfg(feature = "moment")]
pub mod moment;
mod null;
mod to_list;

#[derive(Copy, Clone)]
pub enum NullBehavior {
    /// drop nulls
    Drop,
    /// ignore nulls
    Ignore,
}
