//! Everything you need to get started with Polars.
pub use std::sync::Arc;

pub(crate) use arrow::array::*;
pub use arrow::datatypes::{Field as ArrowField, Schema as ArrowSchema};
pub(crate) use polars_arrow::export::*;
#[cfg(feature = "ewma")]
pub use polars_arrow::kernels::ewm::EWMOptions;
pub use polars_arrow::prelude::*;
pub(crate) use polars_arrow::trusted_len::TrustedLen;

#[cfg(feature = "dtype-categorical")]
pub use crate::chunked_array::logical::categorical::*;
#[cfg(feature = "object")]
pub use crate::chunked_array::object::PolarsObject;
#[cfg(feature = "rolling_window")]
pub use crate::chunked_array::ops::rolling_window::RollingOptionsFixedWindow;
#[cfg(feature = "rank")]
pub use crate::chunked_array::ops::unique::rank::{RankMethod, RankOptions};
#[cfg(feature = "temporal")]
pub use crate::chunked_array::temporal::conversion::*;
pub(crate) use crate::chunked_array::{to_array, ChunkIdIter};
#[cfg(feature = "asof_join")]
pub use crate::frame::asof_join::*;
pub(crate) use crate::frame::{groupby::aggregations::*, hash_join::*};
#[cfg(feature = "checked_arithmetic")]
pub use crate::series::arithmetic::checked::NumOpsDispatchChecked;
pub(crate) use crate::utils::CustomIterTools;
pub use crate::{
    chunked_array::{
        builder::{
            BooleanChunkedBuilder, ChunkedBuilder, ListBooleanChunkedBuilder, ListBuilderTrait,
            ListPrimitiveChunkedBuilder, ListUtf8ChunkedBuilder, NewChunkedArray,
            PrimitiveChunkedBuilder, Utf8ChunkedBuilder,
        },
        iterator::PolarsIterator,
        ops::{aggregate::*, *},
        ChunkedArray,
    },
    datatypes,
    datatypes::*,
    df,
    error::{PolarsError, Result},
    frame::{
        explode::MeltArgs,
        groupby::{GroupsIdx, GroupsProxy, GroupsSlice},
        hash_join::JoinType,
        *,
    },
    named_from::{NamedFrom, NamedFromOwned},
    schema::*,
    series::{
        arithmetic::{LhsNumOps, NumOpsDispatch},
        IntoSeries, Series, SeriesTrait,
    },
    testing::*,
    utils::IntoVec,
    vector_hasher::VecHash,
};
