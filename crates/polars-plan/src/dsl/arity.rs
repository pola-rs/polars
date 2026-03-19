use super::*;

/// Utility struct for the `when-then-otherwise` expression.
///
/// Represents the state of the expression after [when] is called.
///
/// In this state, `then` must be called to continue to finish the expression.
#[derive(Clone)]
pub struct When {
    condition: Expr,
}

/// Utility struct for the `when-then-otherwise` expression.
///
/// Represents the state of the expression after `when(...).then(...)` is called.
#[derive(Clone)]
pub struct Then {
    condition: Expr,
    statement: Expr,
}

/// Utility struct for the `when-then-otherwise` expression.
///
/// Represents the state of the expression after an additional `when` is called.
///
/// In this state, `then` must be called to continue to finish the expression.
#[derive(Clone)]
pub struct ChainedWhen {
    conditions: Vec<Expr>,
    statements: Vec<Expr>,
}

/// Utility struct for the `when-then-otherwise` expression.
///
/// Represents the state of the expression after an additional `then` is called.
#[derive(Clone)]
pub struct ChainedThen {
    conditions: Vec<Expr>,
    statements: Vec<Expr>,
}

impl When {
    /// Add a condition to the `when-then-otherwise` expression.
    pub fn then<E: Into<Expr>>(self, expr: E) -> Then {
        Then {
            condition: self.condition,
            statement: expr.into(),
        }
    }
}

impl Then {
    /// Attach a statement to the corresponding condition.
    pub fn when<E: Into<Expr>>(self, condition: E) -> ChainedWhen {
        ChainedWhen {
            conditions: vec![self.condition, condition.into()],
            statements: vec![self.statement],
        }
    }

    /// Define a default for the `when-then-otherwise` expression.
    pub fn otherwise<E: Into<Expr>>(self, statement: E) -> Expr {
        ternary_expr(self.condition, self.statement, statement.into())
    }
}

impl ChainedWhen {
    pub fn then<E: Into<Expr>>(mut self, statement: E) -> ChainedThen {
        self.statements.push(statement.into());
        ChainedThen {
            conditions: self.conditions,
            statements: self.statements,
        }
    }
}

impl ChainedThen {
    /// Add another condition to the `when-then-otherwise` expression.
    pub fn when<E: Into<Expr>>(mut self, condition: E) -> ChainedWhen {
        self.conditions.push(condition.into());

        ChainedWhen {
            conditions: self.conditions,
            statements: self.statements,
        }
    }

    /// Define a default for the `when-then-otherwise` expression.
    pub fn otherwise<E: Into<Expr>>(self, expr: E) -> Expr {
        // we iterate the preds/ exprs last in first out
        // and nest them.
        //
        // // this expr:
        //   when((col('x') == 'a')).then(1)
        //         .when(col('x') == 'b').then(2)
        //         .when(col('x') == 'c').then(3)
        //         .otherwise(4)
        //
        // needs to become:
        //       when((col('x') == 'a')).then(1)                        -
        //         .otherwise(                                           |
        //             when(col('x') == 'b').then(2)            -        |
        //             .otherwise(                               |       |
        //                 pl.when(col('x') == 'c').then(3)      |       |
        //                 .otherwise(4)                         | inner | outer
        //             )                                         |       |
        //         )                                            _|      _|
        //
        // by iterating LIFO we first create
        // `inner` and then assign that to `otherwise`,
        // which will be used in the next layer `outer`
        //

        let conditions_iter = self.conditions.into_iter().rev();
        let mut statements_iter = self.statements.into_iter().rev();

        let mut otherwise = expr.into();

        for e in conditions_iter {
            otherwise = ternary_expr(
                e,
                statements_iter
                    .next()
                    .expect("expr expected, did you call when().then().otherwise?"),
                otherwise,
            );
        }

        otherwise
    }
}

/// Start a `when-then-otherwise` expression.
pub fn when<E: Into<Expr>>(condition: E) -> When {
    When {
        condition: condition.into(),
    }
}

pub fn ternary_expr(predicate: Expr, truthy: Expr, falsy: Expr) -> Expr {
    Expr::Ternary {
        predicate: Arc::new(predicate),
        truthy: Arc::new(truthy),
        falsy: Arc::new(falsy),
    }
}

/// Compute `op(l, r)` (or equivalently `l op r`). `l` and `r` must have types compatible with the Operator.
pub fn binary_expr(l: Expr, op: Operator, r: Expr) -> Expr {
    Expr::BinaryExpr {
        left: Arc::new(l),
        op,
        right: Arc::new(r),
    }
}
