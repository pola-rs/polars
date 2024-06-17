use super::*;

// Executor are the executors of the physical plan and produce DataFrames. They
// combine physical expressions, which produce Series.

/// Executors will evaluate physical expressions and collect them in a DataFrame.
///
/// Executors have other executors as input. By having a tree of executors we can execute the
/// physical plan until the last executor is evaluated.
pub trait Executor: Send {
    fn execute(&mut self, cache: &mut ExecutionState) -> PolarsResult<DataFrame>;
}

pub struct Dummy {}
impl Executor for Dummy {
    fn execute(&mut self, _cache: &mut ExecutionState) -> PolarsResult<DataFrame> {
        panic!("should not get here");
    }
}

impl Default for Box<dyn Executor> {
    fn default() -> Self {
        Box::new(Dummy {})
    }
}
