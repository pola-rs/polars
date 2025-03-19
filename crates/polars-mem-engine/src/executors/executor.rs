use super::*;

// Executor are the executors of the physical plan and produce DataFrames. They
// combine physical expressions, which produce Series.

/// Executors will evaluate physical expressions and collect them in a DataFrame.
///
/// Executors have other executors as input. By having a tree of executors we can execute the
/// physical plan until the last executor is evaluated.
pub trait Executor: Send + Sync {
    fn execute(&mut self, cache: &mut ExecutionState) -> PolarsResult<DataFrame>;
}

type SinkFn =
    Box<dyn FnMut(DataFrame, &mut ExecutionState) -> PolarsResult<Option<DataFrame>> + Send + Sync>;
pub struct SinkExecutor {
    pub name: String,
    pub input: Box<dyn Executor>,
    pub f: SinkFn,
}

impl Executor for SinkExecutor {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run sink_{}", self.name)
            }
        }
        let df = self.input.execute(state)?;

        let profile_name = if state.has_node_timer() {
            Cow::Owned(format!(".sink_{}()", &self.name))
        } else {
            Cow::Borrowed("")
        };

        state.clone().record(
            || (self.f)(df, state).map(|df| df.unwrap_or_else(DataFrame::empty)),
            profile_name,
        )
    }
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
