//! Everything you need to get started with Polars.
pub use std::sync::Arc;

pub use arrow::datatypes::{ArrowPrimitiveType, Field as ArrowField, Schema as ArrowSchema};

pub use polars_arrow::vec::AlignedVec;
pub(crate) use polars_arrow::{array::*, trusted_len::TrustedLen};

#[cfg(feature = "object")]
pub use crate::chunked_array::object::PolarsObject;
pub use crate::chunked_array::ops::take::take_random::{
    IntoTakeRandom, NumTakeRandomChunked, NumTakeRandomCont,
};
#[cfg(feature = "temporal")]
pub use crate::chunked_array::temporal::conversion::*;
pub(crate) use crate::frame::groupby::aggregations::*;
#[cfg(feature = "checked_arithmetic")]
pub use crate::series::arithmetic::checked::NumOpsDispatchChecked;
pub use crate::{
    chunked_array::{
        arithmetic::Pow,
        builder::{
            BooleanChunkedBuilder, ChunkedBuilder, ListBooleanChunkedBuilder, ListBuilderTrait,
            ListPrimitiveChunkedBuilder, ListUtf8ChunkedBuilder, NewChunkedArray,
            PrimitiveChunkedBuilder, Utf8ChunkedBuilder,
        },
        comparison::NumComp,
        iterator::{IntoNoNullIterator, PolarsIterator},
        ops::{aggregate::*, chunkops::ChunkOps, window::InitFold, *},
        ChunkedArray,
    },
    datatypes,
    datatypes::*,
    error::{PolarsError, Result},
    frame::{hash_join::JoinType, DataFrame},
    series::{
        arithmetic::{LhsNumOps, NumOpsDispatch},
        IntoSeries, NamedFrom, Series, SeriesTrait,
    },
    testing::*,
    utils::IntoVec,
    vector_hasher::VecHash,
};
