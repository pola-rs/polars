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

/// Spawns a blocking task to a background thread.
///
/// This uses `pl_async::get_runtime().spawn_blocking` if the `async` feature enabled. It uses
/// `std::thread::spawn` otherwise.
pub fn spawn_blocking<F, R>(func: F)
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    #[cfg(feature = "async")]
    {
        crate::pl_async::get_runtime().spawn_blocking(func);
    }
    #[cfg(not(feature = "async"))]
    {
        std::thread::spawn(func);
    }
}
