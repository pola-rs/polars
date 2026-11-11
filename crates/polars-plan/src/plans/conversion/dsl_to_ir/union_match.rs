//! Lowers `union.match` to a `when/then` chain over the variants.
use std::sync::Arc;

use super::expr_to_ir::{ExprToIRContext, to_aexpr_impl};
use super::*;
use crate::dsl::functions::lit;
use crate::dsl::{Expr, FunctionExpr, UnionFunction, when};
use crate::plans::visitor::TreeWalker;
use crate::plans::{NULL, is_elementwise_rec};

/// Lowers `input[0].union.match(...)`, where `input` holds the union, then one arm per name in
/// `names`, then the default if there is one.
pub(super) fn lower_union_match(
    input: &[Expr],
    names: &[PlSmallStr],
    has_default: bool,
    ctx: &mut ExprToIRContext,
) -> PolarsResult<(Node, PlSmallStr)> {
    let union = &input[0];
    let arms = &input[1..1 + names.len()];
    let default = has_default.then(|| &input[1 + names.len()]);

    let (union_node, output_name) = to_aexpr_impl(union, ctx)?;
    let dtype = ctx.arena.get(union_node).to_dtype(&ctx.to_field_ctx())?;
    let DataType::Union(fields) = &dtype else {
        polars_bail!(InvalidOperation: "`union.match` requires a Union dtype, got `{dtype}`");
    };
    let mut seen = PlIndexSet::with_capacity(names.len());
    for name in names {
        polars_ensure!(
            seen.insert(name),
            Duplicate: "variant `{name}` has more than one arm in `union.match`"
        );
        fields.try_index_of(name)?;
    }
    if !has_default {
        let missing: Vec<_> = fields
            .iter()
            .filter(|f| !seen.contains(f.name()))
            .map(|f| format!("`{}`", f.name()))
            .collect();
        polars_ensure!(
            missing.is_empty(),
            InvalidOperation: "`union.match` has no arm for the variants [{}]; add arms or a `default`",
            missing.join(", ")
        );
    }

    let branches: Vec<(&PlSmallStr, Expr)> = names
        .iter()
        .zip(arms)
        .map(|(name, arm)| {
            let value = union.clone().union().field(name.clone());
            (name, replace_element(arm.clone(), &value))
        })
        .collect();

    // `when/then` would take the supertype of the arms; require them to agree instead.
    let mut common: Option<(String, DataType)> = None;
    let checked = branches
        .iter()
        .map(|(name, value)| (format!("variant `{name}`"), value))
        .chain(default.map(|d| ("`default`".to_string(), d)));
    for (what, value) in checked {
        let (node, _) = to_aexpr_impl(value, ctx)?;
        polars_ensure!(
            is_elementwise_rec(node, ctx.arena),
            InvalidOperation: "the arm for {what} in `union.match` must be elementwise"
        );
        let dt = ctx.arena.get(node).to_dtype(&ctx.to_field_ctx())?;
        if dt.is_null() || dt.is_unknown() {
            continue;
        }
        match &common {
            None => common = Some((what, dt)),
            Some((first, c)) => polars_ensure!(
                c == &dt,
                SchemaMismatch: "the arms of `union.match` must have the same dtype, \
                but the arm for {first} gives `{c}` and the arm for {what} gives `{dt}`"
            ),
        }
    }

    let tag = union.clone().union().tag();
    // Null rows match no arm and must not take the default.
    let first = when(union.clone().is_null()).then(lit(NULL));
    let default = default.cloned().unwrap_or_else(|| lit(NULL));
    let mut branches = branches.into_iter();
    let out = match branches.next() {
        None => first.otherwise(default),
        Some((name, value)) => {
            let mut chain = first.when(tag.clone().eq(lit(name.as_str()))).then(value);
            for (name, value) in branches {
                chain = chain.when(tag.clone().eq(lit(name.as_str()))).then(value);
            }
            chain.otherwise(default)
        },
    };
    let out = out.alias(output_name);
    let (node, name) = to_aexpr_impl(&out, ctx)?;
    if let Some((first, c)) = common {
        let dt = ctx.arena.get(node).to_dtype(&ctx.to_field_ctx())?;
        polars_ensure!(
            dt == c,
            SchemaMismatch: "the arms of `union.match` must have the same dtype, but the arm for \
            {first} gives `{c}` and a literal arm makes it `{dt}`"
        );
    }
    Ok((node, name))
}

/// Replaces `pl.element()` in `expr` by `with`, except where a nested scope rebinds it.
fn replace_element(expr: Expr, with: &Expr) -> Expr {
    let rec = |e: Arc<Expr>| Arc::new(replace_element(Arc::unwrap_or_clone(e), with));
    match expr {
        Expr::Element => with.clone(),
        Expr::Eval {
            expr,
            evaluation,
            variant,
        } => Expr::Eval {
            expr: rec(expr),
            evaluation,
            variant,
        },
        Expr::Function {
            mut input,
            function: function @ FunctionExpr::UnionExpr(UnionFunction::Match { .. }),
        } => {
            input[0] = replace_element(input[0].clone(), with);
            Expr::Function { input, function }
        },
        e => e
            .map_children(&mut |c, _| Ok(replace_element(c, with)), &mut ())
            .unwrap(),
    }
}
