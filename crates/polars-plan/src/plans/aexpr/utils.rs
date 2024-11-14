use bitflags::bitflags;

use super::*;

fn has_series_or_range(ae: &AExpr) -> bool {
    matches!(
        ae,
        AExpr::Literal(LiteralValue::Series(_) | LiteralValue::Range { .. })
    )
}

bitflags! {
        #[derive(Default, Copy, Clone)]
        struct StreamableFlags: u8 {
          const ALLOW_CAST_CATEGORICAL = 1;
        }
}

#[derive(Copy, Clone)]
pub struct IsStreamableContext {
    flags: StreamableFlags,
    context: Context,
}

impl Default for IsStreamableContext {
    fn default() -> Self {
        Self {
            flags: StreamableFlags::all(),
            context: Default::default(),
        }
    }
}

impl IsStreamableContext {
    pub fn new(ctx: Context) -> Self {
        Self {
            flags: StreamableFlags::all(),
            context: ctx,
        }
    }

    pub fn with_allow_cast_categorical(mut self, allow_cast_categorical: bool) -> Self {
        self.flags.set(
            StreamableFlags::ALLOW_CAST_CATEGORICAL,
            allow_cast_categorical,
        );
        self
    }
}

pub fn is_streamable(node: Node, expr_arena: &Arena<AExpr>, ctx: IsStreamableContext) -> bool {
    // check whether leaf column is Col or Lit
    let mut seen_column = false;
    let mut seen_lit_range = false;
    let all = expr_arena.iter(node).all(|(_, ae)| match ae {
        AExpr::Function {
            function: FunctionExpr::SetSortedFlag(_),
            ..
        } => true,
        AExpr::Function { options, .. } | AExpr::AnonymousFunction { options, .. } => {
            match ctx.context {
                Context::Default => matches!(
                    options.collect_groups,
                    ApplyOptions::ElementWise | ApplyOptions::ApplyList
                ),
                Context::Aggregation => matches!(options.collect_groups, ApplyOptions::ElementWise),
            }
        },
        AExpr::Column(_) => {
            seen_column = true;
            true
        },
        AExpr::BinaryExpr { left, right, .. } => {
            !has_aexpr(*left, expr_arena, has_series_or_range)
                && !has_aexpr(*right, expr_arena, has_series_or_range)
        },
        AExpr::Ternary {
            truthy,
            falsy,
            predicate,
        } => {
            !has_aexpr(*truthy, expr_arena, has_series_or_range)
                && !has_aexpr(*falsy, expr_arena, has_series_or_range)
                && !has_aexpr(*predicate, expr_arena, has_series_or_range)
        },
        #[cfg(feature = "dtype-categorical")]
        AExpr::Cast { dtype, .. } if matches!(dtype, DataType::Categorical(_, _)) => {
            ctx.flags.contains(StreamableFlags::ALLOW_CAST_CATEGORICAL)
        },
        AExpr::Alias(_, _) | AExpr::Cast { .. } => true,
        AExpr::Literal(lv) => match lv {
            LiteralValue::Series(_) | LiteralValue::Range { .. } => {
                seen_lit_range = true;
                true
            },
            _ => true,
        },
        _ => false,
    });

    if all {
        // adding a range or literal series to chunks will fail because sizes don't match
        // if column is a leaf column then it is ok
        // - so we want to block `with_column(lit(Series))`
        // - but we want to allow `with_column(col("foo").is_in(Series))`
        // that means that IFF we seen a lit_range, we only allow if we also seen a `column`.
        return if seen_lit_range { seen_column } else { true };
    }

    false
}

pub fn all_streamable(
    exprs: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    ctx: IsStreamableContext,
) -> bool {
    exprs
        .iter()
        .all(|e| is_streamable(e.node(), expr_arena, ctx))
}
