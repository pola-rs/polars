use std::collections::BTreeSet;

use super::*;

/// Projection in the physical plan is done by selecting an expression per thread.
/// In case of many projections and columns this can be expensive when the expressions are simple
/// column selections. These can be selected on a single thread. The single thread is faster, because
/// the eager selection algorithm hashes the column names, making the projection complexity linear
/// instead of quadratic.
///
/// It is important that this optimization is ran after projection pushdown.
///
/// The schema reported after this optimization is also
pub(super) struct SimpleProjectionAndCollapse {
    /// Keep track of nodes that are already processed when they
    /// can be expensive. Schema materialization can be for instance.
    processed: BTreeSet<Node>,
    eager: bool,
}

impl SimpleProjectionAndCollapse {
    pub(super) fn new(eager: bool) -> Self {
        Self {
            processed: Default::default(),
            eager,
        }
    }
}

impl OptimizationRule for SimpleProjectionAndCollapse {
    fn optimize_plan(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
        node: Node,
    ) -> PolarsResult<Option<IR>> {
        use IR::*;
        let lp = lp_arena.get(node);

        match lp {
            Select { input, expr, .. } => {
                if !matches!(lp_arena.get(*input), ExtContext { .. })
                    && !self.processed.contains(&node)
                {
                    // First check if we can apply the optimization before we allocate.
                    if !expr.iter().all(|e| {
                        matches!(expr_arena.get(e.node()), AExpr::Column(_)) && !e.has_alias()
                    }) {
                        self.processed.insert(node);
                        return Ok(None);
                    }

                    let exprs = expr
                        .iter()
                        .map(|e| e.output_name().clone())
                        .collect::<Vec<_>>();
                    let Some(alp) = IRBuilder::new(*input, expr_arena, lp_arena)
                        .project_simple(exprs.iter().cloned())
                        .ok()
                    else {
                        return Ok(None);
                    };
                    let alp = alp.build();

                    Ok(Some(alp))
                } else {
                    self.processed.insert(node);
                    Ok(None)
                }
            },
            SimpleProjection { columns, input } if !self.eager => {
                match lp_arena.get(*input) {
                    // If there are 2 subsequent fast projections, flatten them and only take the last
                    SimpleProjection {
                        input: prev_input, ..
                    } => Ok(Some(SimpleProjection {
                        input: *prev_input,
                        columns: columns.clone(),
                    })),
                    // Cleanup projections set in projection pushdown just above caches
                    // they are not needed.
                    cache_lp @ Cache { .. } if self.processed.contains(&node) => {
                        let cache_schema = cache_lp.schema(lp_arena);
                        if cache_schema.len() == columns.len()
                            && cache_schema.iter_names().zip(columns.iter_names()).all(
                                |(left_name, right_name)| left_name.as_str() == right_name.as_str(),
                            )
                        {
                            Ok(Some(cache_lp.clone()))
                        } else {
                            Ok(None)
                        }
                    },
                    // If a projection does nothing, remove it.
                    other => {
                        let input_schema = other.schema(lp_arena);
                        // This will fail fast if lengths are not equal
                        if *input_schema.as_ref() == *columns {
                            Ok(Some(other.clone()))
                        } else {
                            self.processed.insert(node);
                            Ok(None)
                        }
                    },
                }
            },
            // if there are 2 subsequent caches, flatten them and only take the inner
            Cache {
                input,
                cache_hits: outer_cache_hits,
                ..
            } if !self.eager => {
                if let Cache {
                    input: prev_input,
                    id,
                    cache_hits,
                } = lp_arena.get(*input)
                {
                    Ok(Some(Cache {
                        input: *prev_input,
                        id: *id,
                        // ensure the counts are updated
                        cache_hits: cache_hits.saturating_add(*outer_cache_hits),
                    }))
                } else {
                    Ok(None)
                }
            },
            _ => Ok(None),
        }
    }
}
