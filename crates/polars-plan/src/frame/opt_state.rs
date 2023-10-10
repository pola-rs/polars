#[derive(Copy, Clone, Debug)]
/// State of the allowed optimizations
pub struct OptState {
    pub projection_pushdown: bool,
    pub predicate_pushdown: bool,
    pub type_coercion: bool,
    pub simplify_expr: bool,
    pub file_caching: bool,
    pub slice_pushdown: bool,
    #[cfg(feature = "cse")]
    pub comm_subplan_elim: bool,
    #[cfg(feature = "cse")]
    pub comm_subexpr_elim: bool,
    pub streaming: bool,
    pub eager: bool,
    pub fast_projection: bool,
}

impl Default for OptState {
    fn default() -> Self {
        OptState {
            projection_pushdown: true,
            predicate_pushdown: true,
            type_coercion: true,
            simplify_expr: true,
            slice_pushdown: true,
            // will be toggled by a scan operation such as csv scan or parquet scan
            file_caching: false,
            #[cfg(feature = "cse")]
            comm_subplan_elim: true,
            #[cfg(feature = "cse")]
            comm_subexpr_elim: true,
            streaming: false,
            fast_projection: true,
            eager: false,
        }
    }
}

/// AllowedOptimizations
pub type AllowedOptimizations = OptState;
