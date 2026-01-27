use std::hash::BuildHasher;

use super::*;

// Utility to cheaply check if we have duplicate sources.
// This may have false positives.
#[cfg(feature = "cse")]
#[derive(Default)]
struct UniqueScans {
    ids: PlHashSet<u64>,
    count: usize,
}

#[cfg(feature = "cse")]
impl UniqueScans {
    fn insert(&mut self, node: Node, lp_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) {
        let alp_node = IRNode::new(node);
        self.ids.insert(
            self.ids
                .hasher()
                .hash_one(alp_node.hashable_and_cmp(lp_arena, expr_arena)),
        );
        self.count += 1;
    }
}

pub(super) struct MemberCollector {
    pub(crate) has_joins_or_unions: bool,
    pub(crate) has_sink_multiple: bool,
    pub(crate) has_cache: bool,
    pub(crate) has_ext_context: bool,
    pub(crate) has_filter_with_join_input: bool,
    pub(crate) has_distinct: bool,
    pub(crate) has_sort: bool,
    pub(crate) has_group_by: bool,
    pub(crate) has_hint: bool,
    pub(crate) with_columns_count: u32,
    #[cfg(feature = "cse")]
    scans: UniqueScans,
}

impl MemberCollector {
    pub(super) fn new() -> Self {
        Self {
            has_joins_or_unions: false,
            has_sink_multiple: false,
            has_cache: false,
            has_ext_context: false,
            has_filter_with_join_input: false,
            has_distinct: false,
            has_sort: false,
            has_group_by: false,
            has_hint: false,
            with_columns_count: 0,
            #[cfg(feature = "cse")]
            scans: UniqueScans::default(),
        }
    }
    pub(super) fn collect(&mut self, root: Node, lp_arena: &Arena<IR>, _expr_arena: &Arena<AExpr>) {
        use IR::*;
        for (_node, alp) in lp_arena.iter(root) {
            match alp {
                SinkMultiple { .. } => self.has_sink_multiple = true,
                Join { .. } | Union { .. } => self.has_joins_or_unions = true,
                Filter { input, .. } => {
                    self.has_filter_with_join_input |= matches!(lp_arena.get(*input), Join { options, .. } if options.args.how.is_cross())
                },
                Distinct { .. } => {
                    self.has_distinct = true;
                },
                GroupBy { .. } => {
                    self.has_group_by = true;
                },
                Sort { .. } => {
                    self.has_sort = true;
                },
                Cache { .. } => self.has_cache = true,
                ExtContext { .. } => self.has_ext_context = true,
                #[cfg(feature = "cse")]
                Scan { .. } => {
                    self.scans.insert(_node, lp_arena, _expr_arena);
                },
                HStack { .. } => {
                    self.with_columns_count += 1;
                },
                HConcat { .. } => {
                    self.has_joins_or_unions = true;
                },
                #[cfg(feature = "cse")]
                DataFrameScan { .. } => {
                    self.scans.insert(_node, lp_arena, _expr_arena);
                },
                #[cfg(all(feature = "cse", feature = "python"))]
                PythonScan { .. } => {
                    self.scans.insert(_node, lp_arena, _expr_arena);
                },
                MapFunction {
                    function: FunctionIR::Hint(_),
                    ..
                } => self.has_hint = true,
                _ => {},
            }
        }
    }

    #[cfg(feature = "cse")]
    pub(super) fn has_duplicate_scans(&self) -> bool {
        self.scans.count != self.scans.ids.len()
    }
}
