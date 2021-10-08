pub(crate) mod take;
#[cfg(any(
    feature = "temporal",
    feature = "dtype-datetime",
    feature = "dtype-date"
))]
#[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
pub mod temporal;
