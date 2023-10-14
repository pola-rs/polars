//! Everything you need to get started with Polars.
pub use std::sync::Arc;

pub(crate) use arrow::array::*;
pub use arrow::datatypes::{Field as ArrowField, Schema as ArrowSchema};
#[cfg(feature = "ewma")]
pub use arrow::legacy::kernels::ewm::EWMOptions;
pub use arrow::legacy::prelude::*;
pub(crate) use arrow::legacy::trusted_len::TrustedLen;
pub(crate) use arrow::util::total_ord::{TotalEq, TotalOrd};

pub use crate::chunked_array::builder::{
    BinaryChunkedBuilder, BooleanChunkedBuilder, ChunkedBuilder, ListBinaryChunkedBuilder,
    ListBooleanChunkedBuilder, ListBuilderTrait, ListPrimitiveChunkedBuilder,
    ListUtf8ChunkedBuilder, NewChunkedArray, PrimitiveChunkedBuilder, Utf8ChunkedBuilder,
};
pub use crate::chunked_array::collect::{ChunkedCollectInferIterExt, ChunkedCollectIterExt};
pub use crate::chunked_array::iterator::PolarsIterator;
#[cfg(feature = "dtype-categorical")]
pub use crate::chunked_array::logical::categorical::*;
#[cfg(feature = "ndarray")]
pub use crate::chunked_array::ndarray::IndexOrder;
#[cfg(feature = "object")]
pub use crate::chunked_array::object::PolarsObject;
pub use crate::chunked_array::ops::aggregate::*;
#[cfg(feature = "rolling_window")]
pub use crate::chunked_array::ops::rolling_window::RollingOptionsFixedWindow;
pub use crate::chunked_array::ops::*;
#[cfg(feature = "temporal")]
pub use crate::chunked_array::temporal::conversion::*;
pub(crate) use crate::chunked_array::ChunkIdIter;
pub use crate::chunked_array::ChunkedArray;
pub use crate::datatypes::{ArrayCollectIterExt, *};
pub use crate::error::{
    polars_bail, polars_ensure, polars_err, polars_warn, PolarsError, PolarsResult,
};
pub use crate::frame::explode::MeltArgs;
#[cfg(feature = "algorithm_group_by")]
pub(crate) use crate::frame::group_by::aggregations::*;
#[cfg(feature = "algorithm_group_by")]
pub use crate::frame::group_by::{GroupsIdx, GroupsProxy, GroupsSlice, IntoGroupsProxy};
pub use crate::frame::{DataFrame, UniqueKeepStrategy};
pub use crate::hashing::{FxHash, VecHash};
pub use crate::named_from::{NamedFrom, NamedFromOwned};
pub use crate::schema::*;
#[cfg(feature = "checked_arithmetic")]
pub use crate::series::arithmetic::checked::NumOpsDispatchChecked;
pub use crate::series::arithmetic::{LhsNumOps, NumOpsDispatch};
pub use crate::series::{IntoSeries, Series, SeriesTrait};
pub use crate::testing::*;
pub(crate) use crate::utils::CustomIterTools;
pub use crate::utils::IntoVec;
pub use crate::{datatypes, df};
