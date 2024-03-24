use super::*;

// ensure the file count counters are decremented with the cache counts
pub(crate) fn decrement_file_counters_by_cache_hits(
    root: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    _expr_arena: &Arena<AExpr>,
    acc_count: FileCount,
    scratch: &mut Vec<Node>,
) {
    use ALogicalPlan::*;
    match lp_arena.get_mut(root) {
        Scan {
            file_options: options,
            ..
        } => {
            if acc_count >= options.file_counter {
                options.file_counter = 1;
            } else {
                options.file_counter -= acc_count as FileCount
            }
        },
        Cache { count, input, .. } => {
            // we use usize::MAX for an infinite cache.
            let new_count = if *count != usize::MAX {
                acc_count + *count as FileCount
            } else {
                acc_count
            };
            decrement_file_counters_by_cache_hits(*input, lp_arena, _expr_arena, new_count, scratch)
        },
        lp => {
            lp.copy_inputs(scratch);
            while let Some(input) = scratch.pop() {
                decrement_file_counters_by_cache_hits(
                    input,
                    lp_arena,
                    _expr_arena,
                    acc_count,
                    scratch,
                )
            }
        },
    }
}