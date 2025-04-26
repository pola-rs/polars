use std::borrow::Cow;
use std::fmt::{Display, Formatter};

use crate::config::verbose;
use crate::format_pl_smallstr;

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
            },
        }
    }
}

pub type Result<T> = std::result::Result<T, PolarsUtilsError>;

#[derive(Debug)]
pub enum TruncateMode {
    /// Truncate from the front, i.e. `...abc`
    Front,
    /// Truncate from the end, i.e. `abc...`
    End,
}

/// Utility whose Display impl truncates the string unless POLARS_VERBOSE is set.
pub struct TruncateErrorDetail<'a> {
    pub content: &'a str,
    pub mode: TruncateMode,
    pub max_length: usize,
}

impl std::fmt::Display for TruncateErrorDetail<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let maybe_truncated = if verbose() {
            self.content
        } else {
            // Clamp the output on non-verbose
            match self.mode {
                TruncateMode::Front => {
                    let start: usize = self.content.len().saturating_sub(self.max_length);
                    &self.content[start..]
                },
                TruncateMode::End => &self.content[..self.content.len().min(self.max_length)],
            }
        };

        if maybe_truncated.len() == self.content.len() {
            f.write_str(self.content)?;
        } else {
            match self.mode {
                TruncateMode::Front => f.write_str(&format!("...{maybe_truncated} "))?,
                TruncateMode::End => f.write_str(&format!("{maybe_truncated}..."))?,
            }

            let n_more = self.content.len() - maybe_truncated.len();
            f.write_str("(set POLARS_VERBOSE=1 to see full response (")?;
            f.write_str(&format_pl_smallstr!("{}", n_more))?;
            f.write_str(" more characters))")?;
        }

        Ok(())
    }
}
