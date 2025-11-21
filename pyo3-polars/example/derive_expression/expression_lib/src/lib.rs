use pyo3_polars::PolarsAllocator;

mod distances;
mod expressions;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
