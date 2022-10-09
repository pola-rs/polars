mod convert;
mod dispatcher;

pub use convert::create_pipeline;
pub use dispatcher::Pipeline;

pub use crate::executors::sinks::groupby::aggregates::can_convert_to_hash_agg;
