pub(crate) mod take;
#[cfg(any(
    feature = "temporal",
    feature = "dtype-date64",
    feature = "dtype-date32"
))]
#[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
pub mod temporal;
