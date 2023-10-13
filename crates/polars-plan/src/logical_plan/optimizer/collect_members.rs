use super::*;

pub(super) struct MemberCollector {
    pub(crate) has_joins_or_unions: bool,
    pub(crate) has_cache: bool,
    pub(crate) has_ext_context: bool,
}

impl MemberCollector {
    pub(super) fn new() -> Self {
        Self {
            has_joins_or_unions: false,
            has_cache: false,
            has_ext_context: false,
        }
    }
    pub fn collect(&mut self, root: Node, lp_arena: &Arena<ALogicalPlan>) {
        use ALogicalPlan::*;
        for (_, alp) in lp_arena.iter(root) {
            match alp {
                Join { .. } | Union { .. } => self.has_joins_or_unions = true,
                Cache { .. } => self.has_cache = true,
                ExtContext { .. } => self.has_ext_context = true,
                _ => {},
            }
        }
    }
}
