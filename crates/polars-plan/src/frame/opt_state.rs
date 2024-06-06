#[derive(Copy, Clone, Debug)]
/// State of the allowed optimizations
pub struct OptState {
    /// Only read columns that are used later in the query.
    pub projection_pushdown: bool,
    /// Apply predicates/filters as early as possible.
    pub predicate_pushdown: bool,
    /// Cluster sequential `with_columns` calls to independent calls.
    pub cluster_with_columns: bool,
    /// Run many type coercion optimization rules until fixed point.
    pub type_coercion: bool,
    /// Run many expression optimization rules until fixed point.
    pub simplify_expr: bool,
    /// Cache file reads.
    pub file_caching: bool,
    /// Pushdown slices/limits.
    pub slice_pushdown: bool,
    #[cfg(feature = "cse")]
    /// Run common-subplan-elimination. This elides duplicate plans and caches their
    /// outputs.
    pub comm_subplan_elim: bool,
    #[cfg(feature = "cse")]
    /// Run common-subexpression-elimination. This elides duplicate expressions and caches their
    /// outputs.
    pub comm_subexpr_elim: bool,
    /// Run nodes that are capably of doing so on the streaming engine.
    pub streaming: bool,
    /// Run every node eagerly. This turns off multi-node optimizations.
    pub eager: bool,
    /// Replace simple projections with a faster inlined projection that skips the expression engine.
    pub fast_projection: bool,
    /// Try to estimate the number of rows so that joins can determine which side to keep in memory.
    pub row_estimate: bool,
}

impl Default for OptState {
    fn default() -> Self {
        OptState {
            projection_pushdown: true,
            predicate_pushdown: true,
            cluster_with_columns: true,
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
            row_estimate: true,
        }
    }
}

/// AllowedOptimizations
pub type AllowedOptimizations = OptState;
