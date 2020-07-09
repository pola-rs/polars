pub use crate::{
    chunked_array::{
        aggregate::Agg,
        apply::Apply,
        builder::{PrimitiveChunkedBuilder, Utf8ChunkedBuilder},
        chunkops::ChunkOps,
        comparison::{CmpOps, NumComp},
        take::{IntoTakeRandom, NumTakeRandomChunked, NumTakeRandomCont, Take, TakeIndex},
        unique::Unique,
        ChunkedArray, Downcast,
    },
    datatypes,
    datatypes::*,
    error::{PolarsError, Result},
    frame::{
        csv::{CsvReader, CsvWriter},
        DataFrame,
    },
    series::{arithmetic::LhsNumOps, NamedFrom, Series},
    testing::*,
};
pub use arrow::datatypes::{ArrowPrimitiveType, Field};

#[cfg(test)]
pub(crate) fn create_df() -> DataFrame {
    let s0 = Series::new("days", [0, 1, 2, 3, 4].as_ref());
    let s1 = Series::new("temp", [22.1, 19.9, 7., 2., 3.].as_ref());
    DataFrame::new(vec![s0, s1]).unwrap()
}
