use polars_utils::format_pl_smallstr;

use super::expr_expansion::{needs_expansion, rewrite_projections};
use super::expr_to_ir::{ExprToIRContext, to_aexpr_impl, to_expr_irs};
use super::*;
use crate::constants::POLARS_REWRITE_INPUT_PREFIX;
use crate::dsl::DslRewriteSource;

/// The converted inputs of the rewrite whose template is being converted.
pub struct RewriteInputs {
    inputs: Vec<ExprIR>,
    used: Vec<bool>,
}

impl RewriteInputs {
    /// Resolve `Expr::RewriteInput(i)`.
    pub(super) fn get(
        &mut self,
        i: u32,
        arena: &mut Arena<AExpr>,
    ) -> PolarsResult<(Node, PlSmallStr)> {
        let idx = i as usize;
        polars_ensure!(
            idx < self.inputs.len(),
            InvalidOperation: "rewrite_input({i}) is out of bounds for a rewrite with {} inputs",
            self.inputs.len()
        );
        let e = &self.inputs[idx];
        // Arena nodes may not be shared between parents, so later uses get a copy.
        let node = if std::mem::replace(&mut self.used[idx], true) {
            deep_clone_ae(e.node(), arena)
        } else {
            e.node()
        };
        Ok((node, e.output_name().clone()))
    }
}

pub(super) fn rewrite_input_column_name(i: usize) -> PlSmallStr {
    format_pl_smallstr!("{POLARS_REWRITE_INPUT_PREFIX}{i}")
}

pub(super) fn convert_dsl_rewrite(
    input: Vec<Expr>,
    source: DslRewriteSource,
    ctx: &mut ExprToIRContext,
) -> PolarsResult<(Node, PlSmallStr)> {
    let name = source.name();

    let inputs = to_expr_irs(input, ctx)?;
    let fields = inputs
        .iter()
        .map(|e| e.field(ctx.schema, ctx.arena))
        .collect::<PolarsResult<Vec<_>>>()?;

    let template = source
        .rewrite(&fields, ctx.schema)
        .map_err(|e| e.context(format!("rewrite '{name}' failed").into()))?;
    validate_template(&template, fields.len(), &name)?;
    let template = expand_template(template, &fields, ctx.schema, &name)?;

    let output_name = inputs
        .first()
        .map_or_else(|| name.clone(), |e| e.output_name().clone());

    let used = vec![false; inputs.len()];
    let prev = ctx.rewrite_inputs.replace(RewriteInputs { inputs, used });
    let result = to_aexpr_impl(template, ctx);
    ctx.rewrite_inputs = prev;
    let (node, _) = result?;

    Ok((node, output_name))
}

fn validate_template(template: &Expr, n_inputs: usize, name: &str) -> PolarsResult<()> {
    let is_rewrite_input = |e: &Expr| matches!(e, Expr::RewriteInput(_));
    for e in template {
        match e {
            Expr::Function {
                function: FunctionExpr::DslRewrite(_),
                ..
            } => polars_bail!(
                InvalidOperation: "rewrite '{name}' returned an expression that contains another rewrite; nested rewrites are not supported yet"
            ),
            Expr::RewriteInput(i) => polars_ensure!(
                (*i as usize) < n_inputs,
                InvalidOperation: "rewrite '{name}' returned rewrite_input({i}), but it only has {n_inputs} inputs"
            ),
            Expr::Eval { evaluation, .. } => polars_ensure!(
                !has_expr(evaluation, is_rewrite_input),
                InvalidOperation: "rewrite '{name}' returned an expression that uses rewrite_input() inside a nested evaluation (e.g. `list.eval`), which is not supported"
            ),
            #[cfg(feature = "dtype-struct")]
            Expr::StructEval { evaluation, .. } => polars_ensure!(
                !evaluation.iter().any(|e| has_expr(e, is_rewrite_input)),
                InvalidOperation: "rewrite '{name}' returned an expression that uses rewrite_input() inside a nested evaluation (e.g. `struct.with_fields`), which is not supported"
            ),
            _ => {},
        }
    }
    Ok(())
}

fn expand_template(
    template: Expr,
    fields: &[Field],
    schema: &Schema,
    name: &str,
) -> PolarsResult<Expr> {
    if !needs_expansion(&template) {
        return Ok(template);
    }

    // Expansion can need the dtype of an input (e.g. for `rewrite_input(0).list.eval(..)`),
    // so inputs temporarily become hidden columns.
    let mut schema = schema.clone();
    for (i, f) in fields.iter().enumerate() {
        schema.insert(rewrite_input_column_name(i), f.dtype.clone());
    }
    let template = template.map_expr(|e| match e {
        Expr::RewriteInput(i) => Expr::Column(rewrite_input_column_name(i as usize)),
        e => e,
    });

    let mut expanded = rewrite_projections(
        vec![template],
        &Default::default(),
        &schema,
        &mut OptFlags::empty(),
    )?;
    polars_ensure!(
        expanded.len() == 1,
        InvalidOperation: "rewrite '{name}' must return a single expression, but it expanded into {} expressions",
        expanded.len()
    );

    Ok(expanded.pop().unwrap().map_expr(|e| match e {
        Expr::Column(c) => match c
            .strip_prefix(POLARS_REWRITE_INPUT_PREFIX)
            .and_then(|i| i.parse().ok())
        {
            Some(i) => Expr::RewriteInput(i),
            None => Expr::Column(c),
        },
        e => e,
    }))
}
