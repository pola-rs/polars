//! contains a wide range of compute operations (e.g.
//! [`arithmetics`], [`aggregate`],
//! [`filter`], [`comparison`], and [`sort`])
//!
//! This module's general design is
//! that each operator has two interfaces, a statically-typed version and a dynamically-typed
//! version.
//! The statically-typed version expects concrete arrays (such as [`PrimitiveArray`](crate::array::PrimitiveArray));
//! the dynamically-typed version expects `&dyn Array` and errors if the the type is not
//! supported.
//! Some dynamically-typed operators have an auxiliary function, `can_*`, that returns
//! true if the operator can be applied to the particular `DataType`.

#[cfg(any(feature = "compute_aggregate", feature = "io_parquet"))]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_aggregate")))]
pub mod aggregate;
#[cfg(feature = "compute_arithmetics")]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_arithmetics")))]
pub mod arithmetics;
pub mod arity;
pub mod arity_assign;
#[cfg(feature = "compute_bitwise")]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_bitwise")))]
pub mod bitwise;
#[cfg(feature = "compute_boolean")]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_boolean")))]
pub mod boolean;
#[cfg(feature = "compute_boolean_kleene")]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_boolean_kleene")))]
pub mod boolean_kleene;
#[cfg(feature = "compute_cast")]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_cast")))]
pub mod cast;
#[cfg(feature = "compute_comparison")]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_comparison")))]
pub mod comparison;
#[cfg(feature = "compute_concatenate")]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_concatenate")))]
pub mod concatenate;
#[cfg(feature = "compute_filter")]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_filter")))]
pub mod filter;
#[cfg(feature = "compute_if_then_else")]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_if_then_else")))]
pub mod if_then_else;
#[cfg(feature = "compute_take")]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_take")))]
pub mod take;
#[cfg(feature = "compute_temporal")]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_temporal")))]
pub mod temporal;
mod utils;
