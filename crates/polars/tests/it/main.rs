#![cfg_attr(feature = "nightly", allow(clippy::result_large_err))] // remove once stable
#![cfg_attr(feature = "nightly", allow(clippy::manual_repeat_n))] // remove once stable
#![cfg_attr(feature = "nightly", allow(clippy::len_zero))] // remove once stable
mod core;
mod io;
mod joins;
#[cfg(feature = "lazy")]
mod lazy;
mod schema;
mod time;

mod arrow;
mod chunks;

pub static FOODS_CSV: &str = "../../examples/datasets/foods1.csv";
