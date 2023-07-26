use super::*;

/// Intermediate state of `when(..).then(..).otherwise(..)` expression.
#[derive(Clone)]
pub struct When {
    predicate: Expr,
}

/// Intermediate state of `when(..).then(..).otherwise(..)` expression.
#[derive(Clone)]
pub struct Then {
    predicate: Expr,
    then: Expr,
}

/// Intermediate state of a chained `when(..).then(..).otherwise(..)` expression.
#[derive(Clone)]
pub struct ChainedWhen {
    predicates: Vec<Expr>,
    thens: Vec<Expr>,
}

/// Intermediate state of a chained `when(..).then(..).otherwise(..)` expression.
#[derive(Clone)]
pub struct ChainedThen {
    predicates: Vec<Expr>,
    thens: Vec<Expr>,
}

impl When {
    pub fn then<E: Into<Expr>>(self, expr: E) -> Then {
        Then {
            predicate: self.predicate,
            then: expr.into(),
        }
    }
}

impl Then {
    pub fn when<E: Into<Expr>>(self, predicate: E) -> ChainedWhen {
        ChainedWhen {
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

impl ChainedWhen {
    pub fn then<E: Into<Expr>>(mut self, expr: E) -> ChainedThen {
        self.thens.push(expr.into());
        ChainedThen {
            predicates: self.predicates,
            thens: self.thens,
        }
    }
}

impl ChainedThen {
    pub fn when<E: Into<Expr>>(mut self, predicate: E) -> ChainedWhen {
        self.predicates.push(predicate.into());

        ChainedWhen {
            predicates: self.predicates,
            thens: self.thens,
        }
    }

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

        let pred_iter = self.predicates.into_iter().rev();
        let mut then_iter = self.thens.into_iter().rev();

        let mut otherwise = expr.into();

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

        otherwise
    }
}

/// Start a `when(..).then(..).otherwise(..)` expression
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
