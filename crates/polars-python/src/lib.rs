#![allow(clippy::nonstandard_macro_braces)] // Needed because clippy does not understand proc macro of PyO3
#![allow(clippy::transmute_undefined_repr)]
#![allow(non_local_definitions)]
#![allow(clippy::too_many_arguments)] // Python functions can have many arguments due to default arguments
#![allow(clippy::disallowed_types)]
#![allow(clippy::useless_conversion)] // Needed for now due to https://github.com/PyO3/pyo3/issues/4828.
#![cfg_attr(
    feature = "allow_unused",
    allow(unused, dead_code, irrefutable_let_patterns)
)] // Maybe be caused by some feature
// combinations

#[cfg(feature = "csv")]
pub mod batched_csv;
#[cfg(feature = "catalog")]
pub mod catalog;
#[cfg(feature = "polars_cloud_client")]
pub mod cloud_client;
#[cfg(feature = "polars_cloud_server")]
pub mod cloud_server;
pub mod conversion;
pub mod dataframe;
pub mod dataset;
pub mod datatypes;
pub mod error;
pub mod exceptions;
pub mod export;
pub mod expr;
pub mod file;
#[cfg(feature = "pymethods")]
pub mod functions;
pub mod interop;
pub mod io;
pub mod lazyframe;
pub mod lazygroupby;
pub mod map;

#[cfg(feature = "object")]
pub mod object;
#[cfg(feature = "object")]
pub mod on_startup;
pub mod prelude;
pub mod py_modules;
pub mod series;
#[cfg(feature = "sql")]
pub mod sql;
pub mod testing;
pub mod timeout;
pub mod utils;

#[cfg(feature = "c_api")]
pub mod c_api;

use crate::conversion::Wrap;

pub type PyDataType = Wrap<polars_core::datatypes::DataType>;
pub type PySchema = Wrap<polars_core::schema::Schema>;
pub use crate::dataframe::PyDataFrame;
pub use crate::expr::PyExpr;
pub use crate::lazyframe::PyLazyFrame;
pub use crate::lazygroupby::PyLazyGroupBy;
pub use crate::series::PySeries;
