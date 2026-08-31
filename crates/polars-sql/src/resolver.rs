use std::sync::{Arc, Once};

use polars_error::PolarsResult;
use polars_lazy::prelude::LazyFrame;
use polars_plan::dsl::{DslPlan, SqlResolver, set_sql_resolver};
use polars_plan::plans::{AExpr, IR};
use polars_utils::arena::Arena;
use polars_utils::pl_str::PlSmallStr;

use crate::context::SQLContext;

struct DslSqlResolver;

impl SqlResolver for DslSqlResolver {
    fn resolve(
        &self,
        query: &str,
        relations: Vec<(PlSmallStr, DslPlan)>,
        lp_arena: &mut Arena<IR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<DslPlan> {
        let mut ctx = SQLContext::new();
        for (name, plan) in relations {
            ctx.register(&name, LazyFrame::from(plan));
        }
        ctx.execute_with_arenas(query, lp_arena, expr_arena)
    }
}

static REGISTERED: Once = Once::new();

/// Make [`DslPlan::SQL`] resolvable during DSL -> IR conversion.
pub fn register_sql_resolver() {
    REGISTERED.call_once(|| set_sql_resolver(Arc::new(DslSqlResolver)));
}
