use super::*;

// Utility to cheaply check if we have duplicate sources.
// This may have false positives.
#[derive(Default)]
struct UniqueScans {
    ids: PlHashSet<u64>,
    count: usize,
}

impl UniqueScans {
    fn insert(&mut self, node: Node, lp_arena: &Arena<ALogicalPlan>, expr_arena: &Arena<AExpr>) {
        let alp_node = unsafe { ALogicalPlanNode::from_raw(node, lp_arena as *const _ as *mut _) };
        self.ids.insert(
            self.ids
                .hasher()
                .hash_one(alp_node.hashable_and_cmp(expr_arena)),
        );
        self.count += 1;
    }
}

pub(super) struct MemberCollector {
    pub(crate) has_joins_or_unions: bool,
    pub(crate) has_cache: bool,
    pub(crate) has_ext_context: bool,
    scans: UniqueScans,
}

impl MemberCollector {
    pub(super) fn new() -> Self {
        Self {
            has_joins_or_unions: false,
            has_cache: false,
            has_ext_context: false,
            scans: UniqueScans::default(),
        }
    }
    pub(super) fn collect(
        &mut self,
        root: Node,
        lp_arena: &Arena<ALogicalPlan>,
        expr_arena: &Arena<AExpr>,
    ) {
        use ALogicalPlan::*;
        for (node, alp) in lp_arena.iter(root) {
            match alp {
                Join { .. } | Union { .. } => self.has_joins_or_unions = true,
                Cache { .. } => self.has_cache = true,
                ExtContext { .. } => self.has_ext_context = true,
                Scan { .. } => {
                    self.scans.insert(node, lp_arena, expr_arena);
                },
                HConcat {..} => {
                    self.has_joins_or_unions = true;
                }
                DataFrameScan { .. } => {
                    self.scans.insert(node, lp_arena, expr_arena);
                },
                _ => {},
            }
        }
    }

    pub(super) fn has_duplicate_scans(&self) -> bool {
        self.scans.count != self.scans.ids.len()
    }
}
