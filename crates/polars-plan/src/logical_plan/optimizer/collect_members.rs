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
    pub(crate) has_cache: bool,
    pub(crate) has_ext_context: bool,
    #[cfg(feature = "cse")]
    scans: UniqueScans,
}

impl MemberCollector {
    pub(super) fn new() -> Self {
        Self {
            has_joins_or_unions: false,
            has_cache: false,
            has_ext_context: false,
            #[cfg(feature = "cse")]
            scans: UniqueScans::default(),
        }
    }
    pub(super) fn collect(&mut self, root: Node, lp_arena: &Arena<IR>, _expr_arena: &Arena<AExpr>) {
        use IR::*;
        for (_node, alp) in lp_arena.iter(root) {
            match alp {
                Join { .. } | Union { .. } => self.has_joins_or_unions = true,
                Cache { .. } => self.has_cache = true,
                ExtContext { .. } => self.has_ext_context = true,
                #[cfg(feature = "cse")]
                Scan { .. } => {
                    self.scans.insert(_node, lp_arena, _expr_arena);
                },
                HConcat { .. } => {
                    self.has_joins_or_unions = true;
                },
                #[cfg(feature = "cse")]
                DataFrameScan { .. } => {
                    self.scans.insert(_node, lp_arena, _expr_arena);
                },
                _ => {},
            }
        }
    }

    #[cfg(feature = "cse")]
    pub(super) fn has_duplicate_scans(&self) -> bool {
        self.scans.count != self.scans.ids.len()
    }
}
