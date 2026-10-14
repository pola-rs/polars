use super::*;

/// Arenas that hold the IR of `DslPlan::IR` nodes created by [`LazyFrame::collect_schema`].
///
/// A `LazyFrame` derived from another one shares its parent's arenas. Converting the derived plan
/// can then reuse the IR node of its input, instead of converting the whole plan again. Without
/// this, calling `collect_schema` after every step of a chain is quadratic in the chain length.
///
/// The arenas are append-only while shared. Anything that mutates IR (optimization, execution)
/// must work on a clone (see [`LazyFrame::get_arenas`]).
#[derive(Default)]
pub(crate) struct CachedArena {
    lp_arena: Arena<IR>,
    expr_arena: Arena<AExpr>,
    /// Size of the arenas when they were created.
    base_len: usize,
}

/// Minimum size of the arenas before we consider starting fresh ones.
const MIN_COMPACT_LEN: usize = 1024;

impl CachedArena {
    pub(crate) fn len(&self) -> usize {
        self.lp_arena.len() + self.expr_arena.len()
    }

    /// Whether the arenas have grown so much that new conversions should start with fresh
    /// arenas.
    ///
    /// Shared arenas also accumulate the nodes of siblings, e.g. when deriving many `LazyFrame`s
    /// from the same parent. This bounds that garbage to a constant factor. Starting fresh costs
    /// a full conversion of the plan, which is amortized by the growth since the last one.
    fn should_compact(&self) -> bool {
        let len = self.len();
        len > MIN_COMPACT_LEN && len > 2 * self.base_len
    }
}

impl LazyFrame {
    pub fn set_cached_arena(&mut self, lp_arena: Arena<IR>, expr_arena: Arena<AExpr>) {
        let mut arenas = CachedArena {
            lp_arena,
            expr_arena,
            base_len: 0,
        };
        arenas.base_len = arenas.len();
        // Don't overwrite arenas that might be shared with other `LazyFrame`s.
        self.cached_arena = Arc::new(Mutex::new(Some(arenas)));
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
            opt_flags: Some(OptFlags::schema_only()),
        };
        Ok(schema)
    }

    /// Get a handle to the schema — a map from column names to data types — of the current
    /// `LazyFrame` computation.
    ///
    /// Returns an `Err` if the logical plan has already encountered an error (i.e., if
    /// `self.collect()` would fail), `Ok` otherwise.
    pub fn collect_schema(&mut self) -> PolarsResult<SchemaRef> {
        // Fast path: the plan was already converted in the cached arenas.
        if let DslPlan::IR {
            node: Some(node),
            version,
            ..
        } = &self.logical_plan
            && let Some(arenas) = self.cached_arena.lock().unwrap().as_ref()
            && arenas.lp_arena.version() == *version
        {
            return Ok(arenas
                .lp_arena
                .get(*node)
                .schema(&arenas.lp_arena)
                .into_owned());
        }

        // Take the arenas out instead of holding the lock during conversion, as conversion can
        // call user code (e.g. `pipe_with_schema`) that collects the schema of a `LazyFrame`
        // sharing these arenas.
        let shared = self.cached_arena.lock().unwrap().take();
        let (mut arenas, put_back) = match shared {
            Some(arenas) if !arenas.should_compact() => (arenas, None),
            shared => (CachedArena::default(), shared),
        };
        let is_fresh = arenas.len() == 0;

        let result = to_alp(
            self.logical_plan.clone(),
            &mut arenas.expr_arena,
            &mut arenas.lp_arena,
            &mut OptFlags::schema_only(),
        );
        let result = result.map(|node| {
            let schema = arenas
                .lp_arena
                .get(node)
                .schema(&arenas.lp_arena)
                .into_owned();
            // Cache the logical plan so that next schema call is cheap.
            self.logical_plan = DslPlan::IR {
                node: Some(node),
                dsl: Arc::new(std::mem::take(&mut self.logical_plan)),
                version: arenas.lp_arena.version(),
                opt_flags: Some(OptFlags::schema_only()),
            };
            schema
        });
        if is_fresh {
            arenas.base_len = arenas.len();
        }

        // Put the arenas back. If we started fresh arenas, or a `LazyFrame` sharing the slot
        // stored arenas in the meantime, we keep ours to ourselves.
        let own = {
            let mut slot = self.cached_arena.lock().unwrap();
            if put_back.is_some() {
                *slot = put_back;
            }
            match &*slot {
                None => {
                    *slot = Some(arenas);
                    None
                },
                Some(_) => Some(arenas),
            }
        };
        if let Some(arenas) = own {
            self.cached_arena = Arc::new(Mutex::new(Some(arenas)));
        }
        result
    }

    /// Get the arenas to convert and optimize the plan in. These are a clone of the cached
    /// arenas, as optimization mutates the arenas in place.
    pub(super) fn get_arenas(&mut self) -> (Arena<IR>, Arena<AExpr>) {
        match self.cached_arena.lock().unwrap().as_mut() {
            Some(arenas) => (arenas.lp_arena.clone(), arenas.expr_arena.clone()),
            None => (Arena::with_capacity(16), Arena::with_capacity(16)),
        }
    }
}
