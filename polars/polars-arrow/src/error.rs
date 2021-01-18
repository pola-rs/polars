use std::borrow::Cow;
use thiserror::Error as ThisError;

type ErrString = Cow<'static, str>;

#[derive(Debug, ThisError)]
pub enum PolarsError {
    #[error(transparent)]
    ArrowError(#[from] arrow::error::ArrowError),
    #[error("{0}")]
    Other(ErrString),
}
