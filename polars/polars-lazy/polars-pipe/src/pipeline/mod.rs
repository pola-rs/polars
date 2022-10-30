mod convert;
mod dispatcher;

pub use convert::{
    create_pipeline, get_sink, get_operator
};
pub use dispatcher::PipeLine;

pub use crate::executors::sinks::groupby::aggregates::can_convert_to_hash_agg;
