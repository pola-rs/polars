use super::*;

pub(crate) struct CachedArena {
    lp_arena: Arena<IR>,
    expr_arena: Arena<AExpr>,
}

impl LazyFrame {
    pub fn set_cached_arena(&self, lp_arena: Arena<IR>, expr_arena: Arena<AExpr>) {
        let mut cached = self.cached_arena.lock().unwrap();
        *cached = Some(CachedArena {
            lp_arena,
            expr_arena,
        });
    }

    pub fn schema_with_arenas(
        &mut self,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<SchemaRef> {
        let node = to_alp(
            self.logical_plan.clone(),
            expr_arena,
            lp_arena,
            &mut OptFlags::schema_only(),
        )?;

        let schema = lp_arena.get(node).schema(lp_arena).into_owned();
        // Cache the logical plan so that next schema call is cheap.
        self.logical_plan = DslPlan::IR {
            node: Some(node),
            dsl: Arc::new(self.logical_plan.clone()),
            version: lp_arena.version(),
        };
        Ok(schema)
    }

    /// Get a handle to the schema — a map from column names to data types — of the current
    /// `LazyFrame` computation.
    ///
    /// Returns an `Err` if the logical plan has already encountered an error (i.e., if
    /// `self.collect()` would fail), `Ok` otherwise.
    pub fn collect_schema(&mut self) -> PolarsResult<SchemaRef> {
        let mut cached_arenas = self.cached_arena.lock().unwrap();

        match &mut *cached_arenas {
            None => {
                let mut lp_arena = Default::default();
                let mut expr_arena = Default::default();
                // Code duplication because of bchk. :(
                let node = to_alp(
                    self.logical_plan.clone(),
                    &mut expr_arena,
                    &mut lp_arena,
                    &mut OptFlags::schema_only(),
                )?;

                let schema = lp_arena.get(node).schema(&lp_arena).into_owned();
                // Cache the logical plan so that next schema call is cheap.
                self.logical_plan = DslPlan::IR {
                    node: Some(node),
                    dsl: Arc::new(self.logical_plan.clone()),
                    version: lp_arena.version(),
                };
                *cached_arenas = Some(CachedArena {
                    lp_arena,
                    expr_arena,
                });

                Ok(schema)
            },
            Some(arenas) => {
                match self.logical_plan {
                    // We have got arenas and don't need to convert the DSL.
                    DslPlan::IR {
                        node: Some(node), ..
                    } => Ok(arenas
                        .lp_arena
                        .get(node)
                        .schema(&arenas.lp_arena)
                        .into_owned()),
                    _ => {
                        // We have got arenas, but still need to convert (parts) of the DSL.
                        // Code duplication because of bchk. :(
                        let node = to_alp(
                            self.logical_plan.clone(),
                            &mut arenas.expr_arena,
                            &mut arenas.lp_arena,
                            &mut OptFlags::schema_only(),
                        )?;

                        let schema = arenas
                            .lp_arena
                            .get(node)
                            .schema(&arenas.lp_arena)
                            .into_owned();
                        // Cache the logical plan so that next schema call is cheap.
                        self.logical_plan = DslPlan::IR {
                            node: Some(node),
                            dsl: Arc::new(self.logical_plan.clone()),
                            version: arenas.lp_arena.version(),
                        };
                        Ok(schema)
                    },
                }
            },
        }
    }

    pub(super) fn get_arenas(&mut self) -> (Arena<IR>, Arena<AExpr>) {
        match self.cached_arena.lock().unwrap().as_mut() {
            Some(arenas) => (arenas.lp_arena.clone(), arenas.expr_arena.clone()),
            None => (Arena::with_capacity(16), Arena::with_capacity(16)),
        }
    }
}
