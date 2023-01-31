use std::borrow::Cow;
use std::fmt::{Display, Formatter};

type ErrString = Cow<'static, str>;

#[derive(Debug)]
pub enum PolarsUtilsError {
    ComputeError(ErrString),
}

impl Display for PolarsUtilsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PolarsUtilsError::ComputeError(s) => {
                let s = s.as_ref();
                write!(f, "{s}")
            }
        }
    }
}

pub type Result<T> = std::result::Result<T, PolarsUtilsError>;
