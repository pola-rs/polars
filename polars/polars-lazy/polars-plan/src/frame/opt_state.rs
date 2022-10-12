#[derive(Copy, Clone)]
/// State of the allowed optimizations
pub struct OptState {
    pub projection_pushdown: bool,
    pub predicate_pushdown: bool,
    pub type_coercion: bool,
    pub simplify_expr: bool,
    pub file_caching: bool,
    pub slice_pushdown: bool,
    #[cfg(feature = "cse")]
    pub common_subplan_elimination: bool,
    pub streaming: bool,
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
            common_subplan_elimination: true,
            streaming: false,
        }
    }
}

/// AllowedOptimizations
pub type AllowedOptimizations = OptState;
