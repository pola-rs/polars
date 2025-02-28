use std::ops::{Deref, DerefMut};

use super::Series;

/// A very thin wrapper around [`Series`] that represents a [`Column`]ized version of [`Series`].
///
/// At the moment this just conditionally tracks where it was created so that materialization
/// problems can be tracked down.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SeriesColumn {
    inner: Series,

    #[cfg(debug_assertions)]
    #[cfg_attr(feature = "serde", serde(skip))]
    materialized_at: Option<std::sync::Arc<std::backtrace::Backtrace>>,
}

impl SeriesColumn {
    #[track_caller]
    pub fn new(series: Series) -> Self {
        Self {
            inner: series,

            #[cfg(debug_assertions)]
            materialized_at: if std::env::var("POLARS_TRACK_SERIES_MATERIALIZATION").as_deref()
                == Ok("1")
            {
                Some(std::sync::Arc::new(
                    std::backtrace::Backtrace::force_capture(),
                ))
            } else {
                None
            },
        }
    }

    pub fn materialized_at(&self) -> Option<&std::backtrace::Backtrace> {
        #[cfg(debug_assertions)]
        {
            self.materialized_at.as_ref().map(|v| v.as_ref())
        }

        #[cfg(not(debug_assertions))]
        None
    }

    pub fn take(self) -> Series {
        self.inner
    }
}

impl From<Series> for SeriesColumn {
    #[track_caller]
    #[inline(always)]
    fn from(value: Series) -> Self {
        Self::new(value)
    }
}

impl Deref for SeriesColumn {
    type Target = Series;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for SeriesColumn {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
