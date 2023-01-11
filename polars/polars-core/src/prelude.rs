//! Everything you need to get started with Polars.
pub use std::sync::Arc;

pub(crate) use arrow::array::*;
pub use arrow::datatypes::{Field as ArrowField, Schema as ArrowSchema};
pub(crate) use polars_arrow::export::*;
#[cfg(feature = "ewma")]
pub use polars_arrow::kernels::ewm::EWMOptions;
pub use polars_arrow::prelude::*;
pub(crate) use polars_arrow::trusted_len::TrustedLen;

#[cfg(feature = "dtype-binary")]
pub use crate::chunked_array::builder::{BinaryChunkedBuilder, ListBinaryChunkedBuilder};
pub use crate::chunked_array::builder::{
    BooleanChunkedBuilder, ChunkedBuilder, ListBooleanChunkedBuilder, ListBuilderTrait,
    ListPrimitiveChunkedBuilder, ListUtf8ChunkedBuilder, NewChunkedArray, PrimitiveChunkedBuilder,
    Utf8ChunkedBuilder,
};
pub use crate::chunked_array::iterator::PolarsIterator;
#[cfg(feature = "dtype-categorical")]
pub use crate::chunked_array::logical::categorical::*;
#[cfg(feature = "object")]
pub use crate::chunked_array::object::PolarsObject;
pub use crate::chunked_array::ops::aggregate::*;
#[cfg(feature = "rolling_window")]
pub use crate::chunked_array::ops::rolling_window::RollingOptionsFixedWindow;
#[cfg(feature = "rank")]
pub use crate::chunked_array::ops::unique::rank::{RankMethod, RankOptions};
pub use crate::chunked_array::ops::*;
#[cfg(feature = "temporal")]
pub use crate::chunked_array::temporal::conversion::*;
pub use crate::chunked_array::ChunkedArray;
pub(crate) use crate::chunked_array::{to_array, ChunkIdIter};
pub use crate::datatypes::*;
pub use crate::error::{PolarsError, PolarsResult};
#[cfg(feature = "asof_join")]
pub use crate::frame::asof_join::*;
pub use crate::frame::explode::MeltArgs;
pub(crate) use crate::frame::groupby::aggregations::*;
pub use crate::frame::groupby::{GroupsIdx, GroupsProxy, GroupsSlice, IntoGroupsProxy};
pub use crate::frame::hash_join::JoinType;
pub(crate) use crate::frame::hash_join::*;
pub use crate::frame::{DataFrame, UniqueKeepStrategy};
pub use crate::named_from::{NamedFrom, NamedFromOwned};
pub use crate::schema::*;
#[cfg(feature = "checked_arithmetic")]
pub use crate::series::arithmetic::checked::NumOpsDispatchChecked;
pub use crate::series::arithmetic::{LhsNumOps, NumOpsDispatch};
pub use crate::series::{IntoSeries, Series, SeriesTrait};
pub use crate::testing::*;
pub(crate) use crate::utils::CustomIterTools;
pub use crate::utils::IntoVec;
pub use crate::vector_hasher::{VecHash, VecHashSingle};
pub use crate::{cloud, datatypes, df};
