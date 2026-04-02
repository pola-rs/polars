use std::sync::{Arc, Mutex};

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_expr::state::ExecutionState;
use polars_mem_engine::Executor;
use polars_plan::dsl::SinkTypeIR;
use polars_plan::plans::{AExpr, IR};
use polars_utils::arena::{Arena, Node};

pub fn build_streaming_query_executor(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<Box<dyn Executor>> {
    let rechunk = match ir_arena.get(node) {
        IR::Scan {
            unified_scan_args, ..
        } => unified_scan_args.rechunk,
        _ => false,
    };

    let node = match ir_arena.get(node) {
        IR::SinkMultiple { .. } => panic!("SinkMultiple not supported"),
        IR::Sink { .. } => node,
        _ => ir_arena.add(IR::Sink {
            input: node,
            payload: SinkTypeIR::Memory,
        }),
    };

    crate::StreamingQuery::build(node, ir_arena, expr_arena)
        .map(Some)
        .map(Mutex::new)
        .map(Arc::new)
        .map(|x| StreamingQueryExecutor {
            executor: x,
            rechunk,
        })
        .map(|x| Box::new(x) as Box<dyn Executor>)
}

// Note: Arc/Mutex is because Executor requires Sync, but SlotMap is not Sync.
struct StreamingQueryExecutor {
    executor: Arc<Mutex<Option<crate::StreamingQuery>>>,
    rechunk: bool,
}

impl Executor for StreamingQueryExecutor {
    fn execute(&mut self, _cache: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let mut df = { self.executor.try_lock().unwrap().take() }
            .expect("unhandled: execute() more than once")
            .execute()
            .map(|x| x.unwrap_single())?;

        if self.rechunk {
            df.rechunk_mut_par();
        }

        Ok(df)
    }
}
