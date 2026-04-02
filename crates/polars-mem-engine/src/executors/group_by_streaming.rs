use std::borrow::Cow;
use std::sync::Arc;

use polars_core::frame::DataFrame;
#[cfg(feature = "dtype-categorical")]
use polars_core::prelude::DataType;
use polars_core::prelude::{Column, GroupsType};
use polars_core::schema::{Schema, SchemaRef};
use polars_core::series::IsSorted;
use polars_error::PolarsResult;
use polars_expr::prelude::PhysicalExpr;
use polars_expr::state::ExecutionState;
use polars_plan::plans::{AExpr, IR, IRPlan};
use polars_utils::arena::{Arena, Node};

use super::{Executor, check_expand_literals, group_by_helper};
use crate::StreamingExecutorBuilder;

pub struct GroupByStreamingExec {
    input_exec: Box<dyn Executor>,
    input_scan_node: Node,
    plan: IRPlan,
    builder: StreamingExecutorBuilder,

    phys_keys: Vec<Arc<dyn PhysicalExpr>>,
    phys_aggs: Vec<Arc<dyn PhysicalExpr>>,
    maintain_order: bool,
    output_schema: SchemaRef,
    slice: Option<(i64, usize)>,
    from_partitioned_ds: bool,
}

impl GroupByStreamingExec {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        input: Box<dyn Executor>,
        builder: StreamingExecutorBuilder,
        root: Node,
        lp_arena: &mut Arena<IR>,
        expr_arena: &Arena<AExpr>,

        phys_keys: Vec<Arc<dyn PhysicalExpr>>,
        phys_aggs: Vec<Arc<dyn PhysicalExpr>>,
        maintain_order: bool,
        output_schema: SchemaRef,
        slice: Option<(i64, usize)>,
        from_partitioned_ds: bool,
    ) -> Self {
        // Create a DataFrame scan for injecting the input result
        let scan = lp_arena.add(IR::DataFrameScan {
            df: Arc::new(DataFrame::empty()),
            schema: Arc::new(Schema::default()),
            output_schema: None,
        });

        let IR::GroupBy {
            input: gb_input, ..
        } = lp_arena.get_mut(root)
        else {
            unreachable!();
        };

        // Set the scan as the group by input
        *gb_input = scan;

        // Prune the subplan into separate arenas
        let mut new_ir_arena = Arena::new();
        let mut new_expr_arena = Arena::new();
        let [new_root, new_scan] = polars_plan::plans::prune::prune(
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
            input_exec: input,
            input_scan_node: new_scan,
            plan,
            builder,
            phys_keys,
            phys_aggs,
            maintain_order,
            output_schema,
            slice,
            from_partitioned_ds,
        }
    }

    fn keys(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Vec<Column>> {
        compute_keys(&self.phys_keys, df, state)
    }
}

fn compute_keys(
    keys: &[Arc<dyn PhysicalExpr>],
    df: &DataFrame,
    state: &ExecutionState,
) -> PolarsResult<Vec<Column>> {
    let evaluated = keys
        .iter()
        .map(|s| s.evaluate(df, state))
        .collect::<PolarsResult<_>>()?;
    let df = check_expand_literals(df, keys, evaluated, false, Default::default())?;
    Ok(df.into_columns())
}

fn estimate_unique_count(keys: &[Column], mut sample_size: usize) -> PolarsResult<usize> {
    // https://stats.stackexchange.com/a/19090/147321
    // estimated unique size
    // u + ui / m (s - m)
    // s: set_size
    // m: sample_size
    // u: total unique groups counted in sample
    // ui: groups with single unique value counted in sample
    let set_size = keys[0].len();
    if set_size < sample_size {
        sample_size = set_size;
    }

    let finish = |groups: &GroupsType| {
        let u = groups.len() as f64;
        let ui = if groups.len() == sample_size {
            u
        } else {
            groups.iter().filter(|g| g.len() == 1).count() as f64
        };

        (u + (ui / sample_size as f64) * (set_size - sample_size) as f64) as usize
    };

    if keys.len() == 1 {
        // we sample as that will work also with sorted data.
        // not that sampling without replacement is *very* expensive. don't do that.
        let s = keys[0].sample_n(sample_size, true, false, None).unwrap();
        // fast multi-threaded way to get unique.
        let groups = s.as_materialized_series().group_tuples(true, false)?;
        Ok(finish(&groups))
    } else {
        let offset = (keys[0].len() / 2) as i64;
        let df = unsafe { DataFrame::new_unchecked_infer_height(keys.to_vec()) };
        let df = df.slice(offset, sample_size);
        let names = df.get_column_names().into_iter().cloned();
        let gb = df.group_by(names).unwrap();
        Ok(finish(gb.get_groups()))
    }
}

// Lower this at debug builds so that we hit this in the test suite.
#[cfg(debug_assertions)]
const PARTITION_LIMIT: usize = 15;
#[cfg(not(debug_assertions))]
const PARTITION_LIMIT: usize = 1000;

// Checks if we should run normal or default aggregation
// by sampling data.
fn can_run_partitioned(
    keys: &[Column],
    original_df: &DataFrame,
    state: &ExecutionState,
    from_partitioned_ds: bool,
) -> PolarsResult<bool> {
    if !keys
        .iter()
        .take(1)
        .all(|s| matches!(s.is_sorted_flag(), IsSorted::Not))
    {
        if state.verbose() {
            eprintln!("FOUND SORTED KEY: running default HASH AGGREGATION")
        }
        Ok(false)
    } else if std::env::var("POLARS_NO_PARTITION").is_ok() {
        if state.verbose() {
            eprintln!("POLARS_NO_PARTITION set: running default HASH AGGREGATION")
        }
        Ok(false)
    } else if std::env::var("POLARS_FORCE_PARTITION").is_ok() {
        if state.verbose() {
            eprintln!("POLARS_FORCE_PARTITION set: running partitioned HASH AGGREGATION")
        }
        Ok(true)
    } else if original_df.height() < PARTITION_LIMIT && !cfg!(test) {
        if state.verbose() {
            eprintln!("DATAFRAME < {PARTITION_LIMIT} rows: running default HASH AGGREGATION")
        }
        Ok(false)
    } else {
        // below this boundary we assume the partitioned group_by will be faster
        let unique_count_boundary = std::env::var("POLARS_PARTITION_UNIQUE_COUNT")
            .map(|s| s.parse::<usize>().unwrap())
            .unwrap_or(1000);

        let (unique_estimate, sampled_method) = match (keys.len(), keys[0].dtype()) {
            #[cfg(feature = "dtype-categorical")]
            (1, DataType::Categorical(_, mapping) | DataType::Enum(_, mapping)) => {
                (mapping.num_cats_upper_bound(), "known")
            },
            _ => {
                // sqrt(N) is a good sample size as it remains low on large numbers
                // it is better than taking a fraction as it saturates
                let sample_size = (original_df.height() as f64).powf(0.5) as usize;

                // we never sample less than 100 data points.
                let sample_size = std::cmp::max(100, sample_size);
                (estimate_unique_count(keys, sample_size)?, "estimated")
            },
        };
        if state.verbose() {
            eprintln!("{sampled_method} unique values: {unique_estimate}");
        }

        if from_partitioned_ds {
            let estimated_cardinality = unique_estimate as f32 / original_df.height() as f32;
            if estimated_cardinality < 0.4 {
                if state.verbose() {
                    eprintln!("PARTITIONED DS");
                }
                Ok(true)
            } else {
                if state.verbose() {
                    eprintln!(
                        "PARTITIONED DS: estimated cardinality: {estimated_cardinality} exceeded the boundary: 0.4, running default HASH AGGREGATION"
                    );
                }
                Ok(false)
            }
        } else if unique_estimate > unique_count_boundary {
            if state.verbose() {
                eprintln!(
                    "estimated unique count: {unique_estimate} exceeded the boundary: {unique_count_boundary}, running default HASH AGGREGATION"
                )
            }
            Ok(false)
        } else {
            Ok(true)
        }
    }
}

impl Executor for GroupByStreamingExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let name = "streaming_group_by";
        state.should_stop()?;
        #[cfg(debug_assertions)]
        {
            if state.verbose() {
                eprintln!("run {name}")
            }
        }
        let input_df = self.input_exec.execute(state)?;

        let profile_name = if state.has_node_timer() {
            Cow::Owned(format!(".{name}()"))
        } else {
            Cow::Borrowed("")
        };

        let keys = self.keys(&input_df, state)?;

        if !can_run_partitioned(&keys, &input_df, state, self.from_partitioned_ds)? {
            return group_by_helper(
                input_df,
                keys,
                &self.phys_aggs,
                None,
                state,
                self.maintain_order,
                &self.output_schema,
                self.slice,
            );
        }

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
