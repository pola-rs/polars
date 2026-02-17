use polars_error::{PolarsResult, polars_err};
use polars_utils::aliases::PlHashMap;

use crate::frame::DataFrame;
use crate::prelude::Column;

pub(super) const LINEAR_SEARCH_LIMIT: usize = 4;

/// Selects columns by name.
pub(super) enum AmortizedColumnSelector<'a> {
    Direct {
        df: &'a DataFrame,
    },
    NameToIdxMapping {
        df: &'a DataFrame,
        name_to_idx: PlHashMap<&'a str, usize>,
    },
}

impl<'a> AmortizedColumnSelector<'a> {
    pub(super) fn new(df: &'a DataFrame) -> Self {
        if df.width() <= LINEAR_SEARCH_LIMIT || df.cached_schema().is_some() {
            Self::Direct { df }
        } else {
            Self::NameToIdxMapping {
                df,
                name_to_idx: df
                    .columns()
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (s.name().as_str(), i))
                    .collect(),
            }
        }
    }

    fn select(&self, name: &str) -> PolarsResult<&'a Column> {
        match self {
            Self::Direct { df } => df.column(name),
            Self::NameToIdxMapping { df, name_to_idx } => {
                let i = *name_to_idx
                    .get(name)
                    .ok_or_else(|| polars_err!(col_not_found = name))?;

                Ok(df.select_at_idx(i).unwrap())
            },
        }
    }

    /// Does not error on duplicate selections.
    pub(super) fn select_multiple(
        &self,
        names: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> PolarsResult<Vec<Column>> {
        names
            .into_iter()
            .map(|name| self.select(name.as_ref()).cloned())
            .collect()
    }
}
