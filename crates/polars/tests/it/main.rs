#![allow(clippy::result_large_err)]
#![allow(clippy::manual_repeat_n)]
#![allow(clippy::len_zero)]
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
