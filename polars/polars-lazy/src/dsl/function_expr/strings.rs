#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum StringFunction {
    Contains {
        pat: String,
        literal: bool,
    },
    StartsWith(String),
    EndsWith(String),
    Extract {
        pat: String,
        group_index: usize,
    },
    #[cfg(feature = "string_justify")]
    Zfill(usize),
    #[cfg(feature = "string_justify")]
    LJust {
        width: usize,
        fillchar: char,
    },
    #[cfg(feature = "string_justify")]
    RJust {
        width: usize,
        fillchar: char,
    },
    ExtractAll(String),
    CountMatch(String),
    #[cfg(feature = "temporal")]
    Strptime(StrpTimeOptions),
    #[cfg(feature = "concat_str")]
    Concat(String),
}

pub(super) fn contains(s: &Series, pat: &str, literal: bool) -> Result<Series> {
    let ca = s.utf8()?;
    if literal {
        ca.contains_literal(pat).map(|ca| ca.into_series())
    } else {
        ca.contains(pat).map(|ca| ca.into_series())
    }
}

pub(super) fn ends_with(s: &Series, sub: &str) -> Result<Series> {
    let ca = s.utf8()?;
    Ok(ca.ends_with(sub).into_series())
}
pub(super) fn starts_with(s: &Series, sub: &str) -> Result<Series> {
    let ca = s.utf8()?;
    Ok(ca.starts_with(sub).into_series())
}

/// Extract a regex pattern from the a string value.
pub(super) fn extract(s: &Series, pat: &str, group_index: usize) -> Result<Series> {
    let pat = pat.to_string();

    let ca = s.utf8()?;
    ca.extract(&pat, group_index).map(|ca| ca.into_series())
}

#[cfg(feature = "string_justify")]
pub(super) fn zfill(s: &Series, alignment: usize) -> Result<Series> {
    let ca = s.utf8()?;
    Ok(ca.zfill(alignment).into_series())
}

#[cfg(feature = "string_justify")]
pub(super) fn ljust(s: &Series, width: usize, fillchar: char) -> Result<Series> {
    let ca = s.utf8()?;
    Ok(ca.ljust(width, fillchar).into_series())
}
#[cfg(feature = "string_justify")]
pub(super) fn rjust(s: &Series, width: usize, fillchar: char) -> Result<Series> {
    let ca = s.utf8()?;
    Ok(ca.rjust(width, fillchar).into_series())
}

pub(super) fn extract_all(s: &Series, pat: &str) -> Result<Series> {
    let pat = pat.to_string();

    let ca = s.utf8()?;
    ca.extract_all(&pat).map(|ca| ca.into_series())
}

pub(super) fn count_match(s: &Series, pat: &str) -> Result<Series> {
    let pat = pat.to_string();

    let ca = s.utf8()?;
    ca.count_match(&pat).map(|ca| ca.into_series())
}

#[cfg(feature = "temporal")]
pub(super) fn strptime(s: &Series, options: &StrpTimeOptions) -> Result<Series> {
    let ca = s.utf8()?;

    let out = match &options.date_dtype {
        DataType::Date => {
            if options.exact {
                ca.as_date(options.fmt.as_deref())?.into_series()
            } else {
                ca.as_date_not_exact(options.fmt.as_deref())?.into_series()
            }
        }
        DataType::Datetime(tu, _) => {
            if options.exact {
                ca.as_datetime(options.fmt.as_deref(), *tu)?.into_series()
            } else {
                ca.as_datetime_not_exact(options.fmt.as_deref(), *tu)?
                    .into_series()
            }
        }
        DataType::Time => {
            if options.exact {
                ca.as_time(options.fmt.as_deref())?.into_series()
            } else {
                return Err(PolarsError::ComputeError(
                    format!("non-exact not implemented for dtype {:?}", DataType::Time).into(),
                ));
            }
        }
        dt => {
            return Err(PolarsError::ComputeError(
                format!("not implemented for dtype {:?}", dt).into(),
            ))
        }
    };
    if options.strict {
        if out.null_count() != ca.null_count() {
            Err(PolarsError::ComputeError(
                "strict conversion to dates failed, maybe set strict=False".into(),
            ))
        } else {
            Ok(out.into_series())
        }
    } else {
        Ok(out.into_series())
    }
}

#[cfg(feature = "concat_str")]
pub(super) fn concat(s: &Series, delimiter: &str) -> Result<Series> {
    Ok(s.str_concat(delimiter).into_series())
}

impl From<StringFunction> for FunctionExpr {
    fn from(str: StringFunction) -> Self {
        FunctionExpr::StringExpr(str)
    }
}
