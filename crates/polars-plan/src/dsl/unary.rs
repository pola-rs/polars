use std::ops::Not;

use crate::dsl::function_expr::BooleanFunction;
use crate::dsl::Expr;

impl Not for Expr {
    type Output = Expr;

    #[allow(clippy::should_implement_trait)]
    fn not(self) -> Expr {
        self.map_private(BooleanFunction::Not.into())
    }
}
