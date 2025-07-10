use std::borrow::Cow;
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_expr::state::ExecutionState;
use polars_plan::dsl::{FileType, SinkTypeIR};
use polars_plan::plans::prune::prune;
use polars_plan::plans::{AExpr, IR, IRPlan};
use polars_utils::arena::{Arena, Node};

use crate::{Executor, StreamingExecutorBuilder};

pub struct PartitionedSinkExecutor {
    name: String,
    input_exec: Box<dyn Executor>,
    input_scan_node: Node,
    plan: IRPlan,
    builder: StreamingExecutorBuilder,
}

impl PartitionedSinkExecutor {
    pub fn new(
        input: Box<dyn Executor>,
        builder: StreamingExecutorBuilder,
        root: Node,
        lp_arena: &mut Arena<IR>,
        expr_arena: &Arena<AExpr>,
    ) -> Self {
        // Create a DataFrame scan for injecting the input result
        let scan = lp_arena.add(IR::DataFrameScan {
            df: Arc::new(DataFrame::empty()),
            schema: Arc::new(Schema::default()),
            output_schema: None,
        });

        let name = {
            let IR::Sink {
                input: sink_input,
                payload: SinkTypeIR::Partition(part),
            } = lp_arena.get_mut(root)
            else {
                unreachable!();
            };

            // Set the scan as the sink input
            *sink_input = scan;

            // Generate a name based on the sink file type
            format!("sink_{}[partitioned]", sink_name(&part.file_type))
        };

        // Prune the subplan into separate arenas
        let mut new_ir_arena = Arena::new();
        let mut new_expr_arena = Arena::new();
        let [new_root, new_scan] = prune(
            &[root, scan],
            lp_arena,
            expr_arena,
            &mut new_ir_arena,
            &mut new_expr_arena,
        )
        .try_into()
        .unwrap();

        let plan = IRPlan {
            lp_top: new_root,
            lp_arena: new_ir_arena,
            expr_arena: new_expr_arena,
        };

        Self {
            name,
            input_exec: input,
            input_scan_node: new_scan,
            plan,
            builder,
        }
    }
}

impl Executor for PartitionedSinkExecutor {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run {}", self.name)
            }
        }
        let input_df = self.input_exec.execute(state)?;

        let profile_name = if state.has_node_timer() {
            Cow::Owned(format!(".{}()", &self.name))
        } else {
            Cow::Borrowed("")
        };

        // Insert the input DataFrame into our DataFrame scan node
        if let IR::DataFrameScan { df, schema, .. } =
            self.plan.lp_arena.get_mut(self.input_scan_node)
        {
            *schema = input_df.schema().clone();
            *df = Arc::new(input_df);
        } else {
            unreachable!();
        }

        let mut streaming_exec = (self.builder)(
            self.plan.lp_top,
            &mut self.plan.lp_arena,
            &mut self.plan.expr_arena,
        )?;

        state
            .clone()
            .record(|| streaming_exec.execute(state), profile_name)
    }
}

pub fn sink_name(file_type: &FileType) -> &'static str {
    match file_type {
        #[cfg(feature = "parquet")]
        FileType::Parquet(_) => "parquet",
        #[cfg(feature = "ipc")]
        FileType::Ipc(_) => "ipc",
        #[cfg(feature = "csv")]
        FileType::Csv(_) => "csv",
        #[cfg(feature = "json")]
        FileType::Json(_) => "json",
        #[allow(unreachable_patterns)]
        _ => panic!("enable filetype feature"),
    }
}
