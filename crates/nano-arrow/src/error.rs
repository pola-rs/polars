//! Defines [`Error`], representing all errors returned by this crate.
use std::fmt::{Debug, Display, Formatter};

/// Enum with all errors in this crate.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Returned when functionality is not yet available.
    NotYetImplemented(String),
    /// Wrapper for an error triggered by a dependency
    External(String, Box<dyn std::error::Error + Send + Sync>),
    /// Wrapper for IO errors
    Io(std::io::Error),
    /// When an invalid argument is passed to a function.
    InvalidArgumentError(String),
    /// Error during import or export to/from a format
    ExternalFormat(String),
    /// Whenever pushing to a container fails because it does not support more entries.
    /// The solution is usually to use a higher-capacity container-backing type.
    Overflow,
    /// Whenever incoming data from the C data interface, IPC or Flight does not fulfil the Arrow specification.
    OutOfSpec(String),
}

impl Error {
    /// Wraps an external error in an `Error`.
    pub fn from_external_error(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::External("".to_string(), Box::new(error))
    }

    pub(crate) fn oos<A: Into<String>>(msg: A) -> Self {
        Self::OutOfSpec(msg.into())
    }

    #[allow(dead_code)]
    pub(crate) fn nyi<A: Into<String>>(msg: A) -> Self {
        Self::NotYetImplemented(msg.into())
    }
}

impl From<::std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::Io(error)
    }
}

impl From<std::str::Utf8Error> for Error {
    fn from(error: std::str::Utf8Error) -> Self {
        Error::External("".to_string(), Box::new(error))
    }
}

impl From<std::string::FromUtf8Error> for Error {
    fn from(error: std::string::FromUtf8Error) -> Self {
        Error::External("".to_string(), Box::new(error))
    }
}

impl From<simdutf8::basic::Utf8Error> for Error {
    fn from(error: simdutf8::basic::Utf8Error) -> Self {
        Error::External("".to_string(), Box::new(error))
    }
}

impl From<std::collections::TryReserveError> for Error {
    fn from(_: std::collections::TryReserveError) -> Error {
        Error::Overflow
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NotYetImplemented(source) => {
                write!(f, "Not yet implemented: {}", &source)
            }
            Error::External(message, source) => {
                write!(f, "External error{}: {}", message, &source)
            }
            Error::Io(desc) => write!(f, "Io error: {desc}"),
            Error::InvalidArgumentError(desc) => {
                write!(f, "Invalid argument error: {desc}")
            }
            Error::ExternalFormat(desc) => {
                write!(f, "External format error: {desc}")
            }
            Error::Overflow => {
                write!(f, "Operation overflew the backing container.")
            }
            Error::OutOfSpec(message) => {
                write!(f, "{message}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Typedef for a [`std::result::Result`] of an [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
