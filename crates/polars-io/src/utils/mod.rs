mod other;

pub use other::*;

pub const URL_ENCODE_CHAR_SET: &percent_encoding::AsciiSet = &percent_encoding::CONTROLS
    .add(b'/')
    .add(b'=')
    .add(b':')
    .add(b' ')
    .add(b'%');
