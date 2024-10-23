use polars_error::PolarsResult;
use crate::datatypes::ListChunked;
use crate::prelude::{ArrayChunked, ChunkFlatten, NamedFrom, Series};

impl ChunkFlatten for ListChunked {
    fn flatten(&self) -> PolarsResult<Series> {
        Ok(
            Series::new(self.name().clone(), self.into_iter().map(|opt_series: Option<Series>| {
                match opt_series.map(|series| series.flatten()) {
                    Some(Ok(series)) => Ok(Some(series)),
                    Some(Err(e)) => Err(e),
                    _ => Ok(None)
                }
            }).collect::<PolarsResult<Vec<Option<Series>>>>()?)
        )
    }
}

impl ChunkFlatten for ArrayChunked {
    fn flatten(&self) -> PolarsResult<Series> {
        Ok(
            Series::new(self.name().clone(), self.into_iter().map(|opt_series: Option<Series>| {
                match opt_series.map(|series| series.flatten()) {
                    Some(Ok(series)) => Ok(Some(series)),
                    Some(Err(e)) => Err(e),
                    _ => Ok(None)
                }
            }).collect::<PolarsResult<Vec<Option<Series>>>>()?)
        )
    }
}