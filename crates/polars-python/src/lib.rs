#![allow(clippy::nonstandard_macro_braces)] // Needed because clippy does not understand proc macro of PyO3
#![allow(clippy::transmute_undefined_repr)]
#![allow(non_local_definitions)]
#![allow(clippy::too_many_arguments)] // Python functions can have many arguments due to default arguments
#![allow(clippy::disallowed_types)]
#![allow(clippy::useless_conversion)] // Needed for now due to https://github.com/PyO3/pyo3/issues/4828.

#[cfg(feature = "csv")]
pub mod batched_csv;
#[cfg(feature = "catalog")]
pub mod catalog;
#[cfg(feature = "polars_cloud")]
pub mod cloud;
pub mod conversion;
pub mod dataframe;
pub mod datatypes;
pub mod error;
pub mod exceptions;
pub mod export;
pub mod expr;
pub mod file;
#[cfg(feature = "pymethods")]
pub mod functions;
pub mod gil_once_cell;
pub mod interop;
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
pub mod timeout;
pub mod utils;

use std::sync::Arc;

use polars::prelude::{DslBuilder, LazyFrame, ParallelStrategy, ScanSources};
use polars_io::HiveOptions;
use polars_utils::mmap::MemSlice;

use crate::conversion::Wrap;
use crate::dataframe::PyDataFrame;
use crate::expr::PyExpr;
use crate::lazyframe::PyLazyFrame;
use crate::lazygroupby::PyLazyGroupBy;
use crate::series::PySeries;

#[test]
fn test() {
    unsafe {
        std::env::set_var("POLARS_MAX_THREADS", "2");
        std::env::set_var("POLARS_FORCE_NEW_STREAMING", "1");
    };

    let dsl = DslBuilder::scan_parquet(
        ScanSources::Buffers(
            (0..13)
                .map(|i| {
                    let bytes = std::fs::read(format!(
                        "/Users/nxs/git/polars/.env/oneshot-deadlock-parquet/{}",
                        i
                    ))
                    .unwrap();
                    MemSlice::from_vec(bytes)
                })
                .collect::<Arc<[_]>>(),
        ),
        None,
        false,
        ParallelStrategy::Auto,
        None,
        false,
        false,
        None,
        false,
        None,
        HiveOptions {
            enabled: Some(false),
            hive_start_idx: 0,
            schema: None,
            try_parse_dates: false,
        },
        false,
        None,
        false,
    )
    .unwrap()
    .slice(99, 89)
    .build();

    let lf = LazyFrame::from(dsl);

    loop {
        lf.clone()
            .collect_with_engine(polars::prelude::Engine::Streaming)
            .unwrap();
    }
}
