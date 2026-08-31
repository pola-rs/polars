use std::sync::{Arc, LazyLock, RwLock};

use polars_error::PolarsResult;
use polars_utils::arena::Arena;
use polars_utils::pl_str::PlSmallStr;

use crate::dsl::DslPlan;
use crate::plans::{AExpr, IR};

/// Resolves a SQL query into a [`DslPlan`].
///
/// Implemented by `polars-sql` and registered through [`set_sql_resolver`], as `polars-plan`
/// cannot depend on `polars-sql`.
pub trait SqlResolver: Send + Sync {
    /// `relations` are the named relations the query may reference. The arenas are those of
    /// the ongoing DSL -> IR conversion.
    fn resolve(
        &self,
        query: &str,
        relations: Vec<(PlSmallStr, DslPlan)>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<DslPlan>;
}

static SQL_RESOLVER: LazyLock<RwLock<Option<Arc<dyn SqlResolver>>>> =
    LazyLock::new(Default::default);

pub fn set_sql_resolver(resolver: Arc<dyn SqlResolver>) {
    let mut lock = SQL_RESOLVER.write().unwrap();
    *lock = Some(resolver);
}

pub fn get_sql_resolver() -> Option<Arc<dyn SqlResolver>> {
    SQL_RESOLVER.read().unwrap().clone()
}
