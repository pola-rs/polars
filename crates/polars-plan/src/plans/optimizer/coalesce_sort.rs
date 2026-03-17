use IR::*;
use polars_core::error::PolarsResult;
use polars_utils::arena::{Arena, Node};

use super::OptimizationRule;
use crate::plans::is_sorted;
use crate::prelude::IR;

pub struct CoalesceSort {}

impl OptimizationRule for CoalesceSort {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut polars_utils::arena::Arena<IR>,
        expr_arena: &mut polars_utils::arena::Arena<crate::prelude::AExpr>,
        node: polars_utils::arena::Node,
    ) -> PolarsResult<Option<IR>> {
        let lp = lp_arena.get(node);

        match lp {
            Sort {
                input,
                by_column,
                slice,
                sort_options,
            } => {
                let input_ir = lp_arena.get(*input);
                let input_sortedness = is_sorted(*input, lp_arena, expr_arena);

                if let IR::Sort {
                    input: in_input,
                    by_column: in_by_column,
                    slice: in_slice,
                    sort_options: in_sort_options,
                } = input_ir
                {
                    // TODO: [amber] If the two sort nodes share a prefix, then
                    // replace these sort nodes with the sort node that sorted
                    // the most columns together
                }

                if let Some(s) = input_sortedness {
                    // TODO: [amber] If the sortedness information starts with the
                    // sortedness requirement of this sort node, then remove the sort node
                }

                Ok(None)
            },
            _ => Ok(None),
        }
    }
}
