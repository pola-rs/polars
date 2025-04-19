pub mod compression;
mod other;

pub use other::*;
#[cfg(feature = "cloud")]
pub mod byte_source;
pub mod file;
pub mod mkdir;
pub mod slice;
pub mod sync_on_close;

pub const URL_ENCODE_CHAR_SET: &percent_encoding::AsciiSet = &percent_encoding::CONTROLS
    .add(b'/')
    .add(b'=')
    .add(b':')
    .add(b' ')
    .add(b'%');
