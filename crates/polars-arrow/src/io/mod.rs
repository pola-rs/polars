#[cfg(feature = "io_ipc")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_ipc")))]
pub mod ipc;

#[cfg(feature = "io_avro")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_avro")))]
pub mod avro;

pub mod iterator;
