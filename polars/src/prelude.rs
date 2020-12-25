//! Everything you need to get started with Polars.
pub use crate::{
    chunked_array::{
        arithmetic::Pow,
        builder::{
            AlignedVec, BooleanChunkedBuilder, ChunkedBuilder, ListBuilderTrait,
            ListPrimitiveChunkedBuilder, ListUtf8ChunkedBuilder, NewChunkedArray,
            PrimitiveChunkedBuilder, Utf8ChunkedBuilder,
        },
        comparison::{CompToSeries, NumComp},
        iterator::{IntoNoNullIterator, PolarsIterator},
        ops::{
            chunkops::ChunkOps,
            take::{AsTakeIndex, IntoTakeRandom, NumTakeRandomChunked, NumTakeRandomCont},
            window::InitFold,
            *,
        },
        ChunkedArray, Downcast, NoNull,
    },
    datatypes,
    datatypes::*,
    error::{PolarsError, Result},
    frame::{
        ser::{
            csv::{CsvReader, CsvWriter},
            SerReader, SerWriter,
        },
        DataFrame, IntoSeries,
    },
    series::{
        arithmetic::{LhsNumOps, NumOpsDispatch},
        NamedFrom, Series, SeriesTrait,
    },
    testing::*,
};
pub use arrow::datatypes::{ArrowPrimitiveType, Field, Schema};
pub use std::sync::Arc;

#[cfg(feature = "temporal")]
pub use crate::chunked_array::temporal::conversion::*;

#[cfg(test)]
pub(crate) fn create_df() -> DataFrame {
    let s0 = Series::new("days", [0, 1, 2, 3, 4].as_ref());
    let s1 = Series::new("temp", [22.1, 19.9, 7., 2., 3.].as_ref());
    DataFrame::new(vec![s0, s1]).unwrap()
}
#[cfg(feature = "parquet")]
pub use crate::frame::ser::parquet::ParquetReader;

#[cfg(feature = "ipc")]
pub use crate::frame::ser::ipc::{IPCReader, IPCWriter};

#[cfg(feature = "json")]
pub use crate::frame::ser::json::JsonReader;

#[cfg(feature = "lazy")]
pub use crate::lazy::frame::*;

#[macro_export]
macro_rules! as_result {
    ($block:block) => {{
        let res: Result<_> = $block;
        res
    }};
}
