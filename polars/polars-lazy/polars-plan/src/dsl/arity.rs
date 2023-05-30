use super::*;

/// Intermediate state of `when(..).then(..).otherwise(..)` expr.
#[derive(Clone)]
pub struct When {
    predicate: Expr,
}

/// Intermediate state of `when(..).then(..).otherwise(..)` expr.
#[derive(Clone)]
pub struct WhenThen {
    predicate: Expr,
    then: Expr,
}

/// Intermediate state of chain when then exprs.
///
/// ```text
/// when(..).then(..)
/// when(..).then(..)
/// when(..).then(..)
/// .otherwise(..)`
/// ```
#[derive(Clone)]
#[must_use]
pub struct WhenThenThen {
    predicates: Vec<Expr>,
    thens: Vec<Expr>,
}

impl When {
    pub fn then<E: Into<Expr>>(self, expr: E) -> WhenThen {
        WhenThen {
            predicate: self.predicate,
            then: expr.into(),
        }
    }
}

impl WhenThen {
    pub fn when<E: Into<Expr>>(self, predicate: E) -> WhenThenThen {
        WhenThenThen {
            predicates: vec![self.predicate, predicate.into()],
            thens: vec![self.then],
        }
    }

    pub fn otherwise<E: Into<Expr>>(self, expr: E) -> Expr {
        Expr::Ternary {
            predicate: Box::new(self.predicate),
            truthy: Box::new(self.then),
            falsy: Box::new(expr.into()),
        }
    }
}

impl WhenThenThen {
    pub fn then(mut self, expr: Expr) -> Self {
        self.thens.push(expr);
        self
    }

    pub fn when(mut self, predicate: Expr) -> Self {
        self.predicates.push(predicate);
        self
    }

    pub fn otherwise(self, expr: Expr) -> Expr {
        // we iterate the preds/ exprs last in first out
        // and nest them.
        //
        // // this expr:
        //   when((col('x') == 'a')).then(1)
        //         .when(col('x') == 'a').then(2)
        //         .when(col('x') == 'b').then(3)
        //         .otherwise(4)
        //
        // needs to become:
        //       when((col('x') == 'a')).then(1)                        -
        //         .otherwise(                                           |
        //             when(col('x') == 'a').then(2)            -        |
        //             .otherwise(                               |       |
        //                 pl.when(col('x') == 'b').then(3)      |       |
        //                 .otherwise(4)                         | inner | outer
        //             )                                         |       |
        //         )                                            _|      _|
        //
        // by iterating lifo we first create
        // `inner` and then assign that to `otherwise`,
        // which will be used in the next layer `outer`
        //

        let pred_iter = self.predicates.into_iter().rev();
        let mut then_iter = self.thens.into_iter().rev();

        let mut otherwise = expr;

        for e in pred_iter {
            otherwise = Expr::Ternary {
                predicate: Box::new(e),
                truthy: Box::new(
                    then_iter
                        .next()
                        .expect("expr expected, did you call when().then().otherwise?"),
                ),
                falsy: Box::new(otherwise),
            }
        }
        if then_iter.next().is_some() {
            panic!(
                "this expr is not properly constructed. \
            Every `when` should have an accompanied `then` call."
            )
        }
        otherwise
    }
}

/// Start a when-then-otherwise expression
pub fn when<E: Into<Expr>>(predicate: E) -> When {
    When {
        predicate: predicate.into(),
    }
}

pub fn ternary_expr(predicate: Expr, truthy: Expr, falsy: Expr) -> Expr {
    Expr::Ternary {
        predicate: Box::new(predicate),
        truthy: Box::new(truthy),
        falsy: Box::new(falsy),
    }
}

/// Compute `op(l, r)` (or equivalently `l op r`). `l` and `r` must have types compatible with the Operator.
pub fn binary_expr(l: Expr, op: Operator, r: Expr) -> Expr {
    Expr::BinaryExpr {
        left: Box::new(l),
        op,
        right: Box::new(r),
    }
}
