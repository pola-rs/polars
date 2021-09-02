#[cfg(feature = "diff")]
pub mod diff;
#[cfg(feature = "moment")]
pub mod moment;

#[derive(Copy, Clone)]
pub enum NullBehavior {
    /// drop nulls
    Drop,
    /// ignore nulls
    Ignore,
}
