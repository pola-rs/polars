use polars_error::PolarsResult;
use crate::datatypes::{DataType, ListChunked};
use crate::prelude::{ArrayChunked, ChunkFlatten, NamedFrom, Series};

impl ChunkFlatten for ListChunked {
    fn flatten(&self) -> PolarsResult<Series> {
        Ok(
            Series::new(self.name().clone(), self.into_iter().map(|opt_series: Option<Series>| {
                opt_series.map(|series| {
                    match series.dtype() {
                        // If it is either a list or an array, we want to flatten the inner data structure, not this one
                        DataType::List(inner) if inner.is_list() || inner.is_array() => series.list().unwrap().flatten(),
                        DataType::Array(inner, _) if inner.is_list() || inner.is_array() => series.array().unwrap().flatten(),
                        _ => series.explode()
                    }.unwrap()
                })
            }).collect::<Vec<Option<Series>>>())
        )
    }
}

impl ChunkFlatten for ArrayChunked {
    fn flatten(&self) -> PolarsResult<Series> {
        Ok(
            Series::new(self.name().clone(), self.into_iter().map(|opt_series: Option<Series>| {
                opt_series.map(|series| {
                    match series.dtype() {
                        DataType::List(_) => series.list().unwrap().flatten(),
                        DataType::Array(_, _) => series.array().unwrap().flatten(),
                        _ => series.explode()
                    }.unwrap()
                })
            }).collect::<Vec<Option<Series>>>())
        )
    }
}