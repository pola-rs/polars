use pyo3_polars::PolarsAllocator;

mod distances;
mod expressions;
mod rewrites;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
