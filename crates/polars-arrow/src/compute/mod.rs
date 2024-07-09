//! contains a wide range of compute operations (e.g.
//! [`arithmetics`], [`aggregate`],
//! [`filter`], [`comparison`], and [`sort`])
//!
//! This module's general design is
//! that each operator has two interfaces, a statically-typed version and a dynamically-typed
//! version.
//! The statically-typed version expects concrete arrays (such as [`PrimitiveArray`](crate::array::PrimitiveArray));
//! the dynamically-typed version expects `&dyn Array` and errors if the type is not
//! supported.
//! Some dynamically-typed operators have an auxiliary function, `can_*`, that returns
//! true if the operator can be applied to the particular `DataType`.

#[cfg(feature = "compute_aggregate")]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_aggregate")))]
pub mod aggregate;
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
pub mod concatenate;
#[cfg(feature = "dtype-decimal")]
pub mod decimal;
#[cfg(feature = "compute_take")]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_take")))]
pub mod take;
#[cfg(feature = "compute_temporal")]
#[cfg_attr(docsrs, doc(cfg(feature = "compute_temporal")))]
pub mod temporal;
pub mod utils;
