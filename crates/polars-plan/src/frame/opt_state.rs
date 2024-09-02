use bitflags::bitflags;

bitflags! {
#[derive(Copy, Clone, Debug)]
    /// Allowed optimizations.
    pub struct OptFlags: u32 {
        /// Only read columns that are used later in the query.
        const PROJECTION_PUSHDOWN = 1;
        /// Apply predicates/filters as early as possible.
        const PREDICATE_PUSHDOWN = 1 << 2;
        /// Cluster sequential `with_columns` calls to independent calls.
        const CLUSTER_WITH_COLUMNS = 1 << 3;
        /// Run many type coercion optimization rules until fixed point.
        const TYPE_COERCION = 1 << 4;
        /// Run many expression optimization rules until fixed point.
        const SIMPLIFY_EXPR = 1 << 5;
        /// Cache file reads.
        const FILE_CACHING = 1 << 6;
        /// Pushdown slices/limits.
        const SLICE_PUSHDOWN = 1 << 7;
        /// Run common-subplan-elimination. This elides duplicate plans and caches their
        /// outputs.
        const COMM_SUBPLAN_ELIM = 1 << 8;
        /// Run common-subexpression-elimination. This elides duplicate expressions and caches their
        /// outputs.
        const COMM_SUBEXPR_ELIM = 1 << 9;
        /// Run nodes that are capably of doing so on the streaming engine.
        const STREAMING = 1 << 10;
        const NEW_STREAMING = 1 << 11;
        /// Run every node eagerly. This turns off multi-node optimizations.
        const EAGER = 1 << 12;
        /// Try to estimate the number of rows so that joins can determine which side to keep in memory.
        const ROW_ESTIMATE = 1 << 13;
        /// Replace simple projections with a faster inlined projection that skips the expression engine.
        const FAST_PROJECTION = 1 << 14;
    }
}

impl OptFlags {
    pub fn schema_only() -> Self {
        Self::TYPE_COERCION
    }
}

impl Default for OptFlags {
    fn default() -> Self {
        Self::from_bits_truncate(u32::MAX) & !Self::NEW_STREAMING & !Self::STREAMING & !Self::EAGER
            // will be toggled by a scan operation such as csv scan or parquet scan
            & !Self::FILE_CACHING
    }
}

/// AllowedOptimizations
pub type AllowedOptimizations = OptFlags;
