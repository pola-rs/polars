//! Lowers `union.match` to [`AExpr::UnionMatch`].
use super::expr_to_ir::{ExprToIRContext, to_aexpr_impl};
use super::*;
use crate::plans::{is_elementwise_rec, union_match_dtype};

/// Lowers `input[0].union.match(...)`, where `input` holds the union, then one arm per name in
/// `names`, then the default if there is one.
pub(super) fn lower_union_match(
    input: &[Expr],
    names: &[PlSmallStr],
    has_default: bool,
    ctx: &mut ExprToIRContext,
) -> PolarsResult<(Node, PlSmallStr)> {
    let arms = &input[1..1 + names.len()];
    let default = has_default.then(|| &input[1 + names.len()]);

    let (expr, output_name) = to_aexpr_impl(&input[0], ctx)?;
    let dtype = ctx.arena.get(expr).to_dtype(&ctx.to_field_ctx())?;
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

    let mut dtypes = Vec::with_capacity(input.len() - 1);
    let mut lower = |e: &Expr, what: String, schema: &Schema, ctx: &mut ExprToIRContext| {
        let mut branch_ctx = ExprToIRContext {
            with_fields: ctx.with_fields.clone(),
            schema,
            arena: ctx.arena,
            allow_unknown: ctx.allow_unknown,
            check_column_names: ctx.check_column_names,
            eval_columns: ctx.eval_columns.take(),
        };
        let node = to_aexpr_impl(e, &mut branch_ctx);
        ctx.eval_columns = branch_ctx.eval_columns;
        let (node, _) = node?;
        polars_ensure!(
            is_elementwise_rec(node, ctx.arena),
            InvalidOperation: "the arm for {what} in `union.match` must be elementwise"
        );
        let dt = ctx
            .arena
            .get(node)
            .to_dtype(&ToFieldContext::new(ctx.arena, schema))?;
        dtypes.push((what, dt));
        PolarsResult::Ok(node)
    };

    let arms = names
        .iter()
        .zip(arms)
        .map(|(name, arm)| {
            let mut schema = ctx.schema.clone();
            let variant = fields[fields.try_index_of(name)?].dtype().clone();
            schema.insert(get_pl_element_name(), variant);
            lower(arm, format!("variant `{name}`"), &schema, ctx)
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    let schema = ctx.schema;
    let default = default
        .map(|d| lower(d, "`default`".to_string(), schema, ctx))
        .transpose()?;
    union_match_dtype(&dtypes)?;

    let node = ctx.arena.add(AExpr::UnionMatch {
        expr,
        names: names.to_vec(),
        arms,
        default,
    });
    Ok((node, output_name))
}
