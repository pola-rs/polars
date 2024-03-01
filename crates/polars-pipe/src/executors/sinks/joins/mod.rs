#[cfg(feature = "cross_join")]
mod cross;
mod generic_build;
mod generic_probe_inner_left;

#[cfg(feature = "cross_join")]
pub(crate) use cross::*;
pub(crate) use generic_build::GenericBuild;
use polars_ops::prelude::JoinType;
