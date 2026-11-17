//! Lambda arguments of list functions: `list_transform(l, lambda x: x + 1)`.
//!
//! sqlparser has no `lambda x: body` syntax in the dialect we use, so the tokens are rewritten
//! into a named argument (`"<marker>x" => body`) before parsing. After parsing, the uses of the
//! lambda parameters in the body are replaced by placeholders that stand for the list element.
use std::ops::ControlFlow;

use polars_core::prelude::{PolarsResult, polars_bail, polars_ensure};
use polars_plan::dsl::Expr;
use polars_plan::dsl::functions::col;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;
use sqlparser::ast::{
    AccessExpr, Expr as SQLExpr, FunctionArg, FunctionArgExpr, FunctionArguments, Ident,
    Value as SQLValue, VisitMut, VisitorMut,
};
use sqlparser::keywords::Keyword;
use sqlparser::tokenizer::{Token, TokenWithSpan, Whitespace, Word};

const LAMBDA_MARKER: &str = "\u{0}lambda\u{0}";
const RESOLVED_LAMBDA_MARKER: &str = "\u{0}lambda resolved\u{0}";
const ELEMENT_MARKER: &str = "\u{0}element\u{0}";

/// Rewrite each `lambda p1, p2: body` into the named argument `"<marker>p1<marker>p2" => body`.
/// Returns whether there was a lambda.
pub(crate) fn rewrite_lambda_tokens(tokens: &mut Vec<TokenWithSpan>) -> bool {
    let mut has_lambdas = false;
    let next_token = |tokens: &[TokenWithSpan], from: usize| {
        (from..tokens.len()).find(|&i| !matches!(tokens[i].token, Token::Whitespace(_)))
    };
    let mut i = 0;
    while i < tokens.len() {
        let is_lambda = matches!(
            &tokens[i].token,
            Token::Word(Word {
                keyword: Keyword::LAMBDA,
                quote_style: None,
                ..
            })
        );
        if !is_lambda {
            i += 1;
            continue;
        }
        let mut params = vec![];
        let mut j = i + 1;
        let colon = loop {
            let Some(p) = next_token(tokens, j) else {
                break None;
            };
            let Token::Word(word) = &tokens[p].token else {
                break None;
            };
            params.push((p, word.value.clone()));
            let Some(sep) = next_token(tokens, p + 1) else {
                break None;
            };
            match &tokens[sep].token {
                Token::Comma => j = sep + 1,
                Token::Colon => break Some((sep, None)),
                // `lambda x:x + 1` reads `:x` as a placeholder.
                Token::Placeholder(s) if s.starts_with(':') && s.len() > 1 => {
                    break Some((sep, Some(s[1..].to_string())));
                },
                _ => break None,
            }
        };
        let Some((colon, body_start)) = colon else {
            i += 1;
            continue;
        };
        let mut name = String::new();
        for (p, param) in &params {
            name.push_str(LAMBDA_MARKER);
            name.push_str(param);
            tokens[*p].token = Token::Whitespace(Whitespace::Space);
        }
        for token in &mut tokens[i + 1..colon] {
            if matches!(token.token, Token::Comma) {
                token.token = Token::Whitespace(Whitespace::Space);
            }
        }
        tokens[i].token = Token::Word(Word {
            value: name,
            quote_style: Some('"'),
            keyword: Keyword::NoKeyword,
        });
        tokens[colon].token = Token::RArrow;
        if let Some(word) = body_start {
            let span = tokens[colon].span;
            tokens.insert(
                colon + 1,
                TokenWithSpan::new(Token::make_word(&word, None), span),
            );
        }
        has_lambdas = true;
        i = colon + 1;
    }
    has_lambdas
}

/// The parameters of a lambda argument named `name`, or `None` if `name` is not a lambda.
fn lambda_params(name: &str) -> Option<Vec<&str>> {
    let params = name.strip_prefix(LAMBDA_MARKER)?;
    Some(params.split(LAMBDA_MARKER).collect())
}

/// The lambda of a function argument, as the number of parameters and the body.
pub(crate) fn lambda_arg(arg: &FunctionArg) -> Option<(usize, &SQLExpr)> {
    let FunctionArg::Named {
        name,
        arg: FunctionArgExpr::Expr(body),
        ..
    } = arg
    else {
        return None;
    };
    let n_params = name
        .value
        .strip_prefix(RESOLVED_LAMBDA_MARKER)?
        .parse()
        .ok()?;
    Some((n_params, body))
}

pub(crate) fn is_lambda_arg(arg: &FunctionArg) -> bool {
    matches!(arg, FunctionArg::Named { name, .. }
        if name.value.starts_with(RESOLVED_LAMBDA_MARKER) || name.value.starts_with(LAMBDA_MARKER))
}

/// Replace the uses of lambda parameters in the lambda bodies of `node` by element
/// placeholders.
pub(crate) fn resolve_lambda_params<T: VisitMut>(node: &mut T) {
    let _ = node.visit(&mut LambdaResolver { scopes: vec![] });
}

struct LambdaResolver {
    /// The parameters of the lambdas around the expression being visited, innermost last.
    scopes: Vec<Vec<String>>,
}

impl LambdaResolver {
    fn element_marker(&self, name: &str) -> Option<SQLExpr> {
        let depth = self
            .scopes
            .iter()
            .rposition(|params| params.iter().any(|p| p == name))?;
        Some(SQLExpr::value(SQLValue::Placeholder(format!(
            "{ELEMENT_MARKER}{}",
            depth + 1
        ))))
    }
}

impl VisitorMut for LambdaResolver {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<()> {
        match expr {
            SQLExpr::Identifier(ident) => {
                if let Some(marker) = self.element_marker(&ident.value) {
                    *expr = marker;
                }
            },
            SQLExpr::CompoundIdentifier(idents) => {
                if let Some(marker) = self.element_marker(&idents[0].value) {
                    let access_chain = idents[1..]
                        .iter()
                        .map(|ident| AccessExpr::Dot(SQLExpr::Identifier(ident.clone())))
                        .collect();
                    *expr = SQLExpr::CompoundFieldAccess {
                        root: Box::new(marker),
                        access_chain,
                    };
                }
            },
            SQLExpr::Function(func) => {
                let FunctionArguments::List(list) = &mut func.args else {
                    return ControlFlow::Continue(());
                };
                for arg in &mut list.args {
                    let FunctionArg::Named { name, arg, .. } = arg else {
                        continue;
                    };
                    let Some(params) = lambda_params(&name.value) else {
                        continue;
                    };
                    self.scopes
                        .push(params.into_iter().map(str::to_string).collect());
                    let _ = arg.visit(self);
                    let n_params = self.scopes.pop().unwrap().len();
                    *name = Ident::with_quote('"', format!("{RESOLVED_LAMBDA_MARKER}{n_params}"));
                }
            },
            _ => {},
        }
        ControlFlow::Continue(())
    }
}

/// The lambda nesting depth of the element that `placeholder` stands for.
pub(crate) fn element_depth(placeholder: &str) -> Option<usize> {
    placeholder.strip_prefix(ELEMENT_MARKER)?.parse().ok()
}

/// The column that stands for the element of the lambda at `depth`, used in the body of the
/// lambda at `lambda_depth`.
pub(crate) fn lambda_element(depth: usize, lambda_depth: usize) -> PolarsResult<Expr> {
    polars_ensure!(
        depth == lambda_depth,
        SQLInterface: "a nested lambda cannot use the parameter of an outer lambda"
    );
    Ok(col(lambda_element_name(depth)))
}

pub(crate) fn lambda_element_name(depth: usize) -> PlSmallStr {
    format_pl_smallstr!("__POLARS_LAMBDA_ELEMENT_{depth}")
}

pub(crate) fn check_lambda_body_expr(expr: &SQLExpr) -> PolarsResult<()> {
    match expr {
        SQLExpr::Identifier(_) | SQLExpr::CompoundIdentifier(_) => {
            polars_bail!(SQLInterface: "a lambda can only use its parameter, found '{}'", expr)
        },
        SQLExpr::Subquery(_) | SQLExpr::Exists { .. } | SQLExpr::InSubquery { .. } => {
            polars_bail!(SQLInterface: "subqueries are not supported in a lambda")
        },
        _ => Ok(()),
    }
}
