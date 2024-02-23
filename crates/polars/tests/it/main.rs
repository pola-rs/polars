mod core;
mod io;
mod joins;
#[cfg(feature = "lazy")]
mod lazy;
mod schema;
mod time;

mod arrow;

pub static FOODS_CSV: &str = "../../examples/datasets/foods1.csv";
