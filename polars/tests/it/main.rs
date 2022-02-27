mod core;
mod io;
mod joins;
#[cfg(feature = "lazy")]
mod lazy;
mod schema;

pub static FOODS_CSV: &str = "../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv";
