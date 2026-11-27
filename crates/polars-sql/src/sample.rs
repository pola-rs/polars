//! Sampling a table: `TABLESAMPLE (10 PERCENT)`, `TABLESAMPLE BERNOULLI (10) REPEATABLE (42)`,
//! `TABLESAMPLE 10%` and `USING SAMPLE 10% (bernoulli, 42)` after the FROM clause.
//!
//! The sample clauses are rewritten into the form sqlparser parses
//! (`TABLESAMPLE (n PERCENT|ROWS) [REPEATABLE (seed)]`), and a `USING SAMPLE` clause is taken
//! out and put on the one table of its FROM clause after parsing.
use std::ops::ControlFlow;

use polars_core::prelude::*;
use polars_lazy::frame::LazyFrame;
use polars_plan::dsl::functions::{col, lit};
use polars_utils::pl_str::PlSmallStr;
use sqlparser::ast::{
    Expr as SQLExpr, Query, SetExpr, TableFactor, TableSample, TableSampleKind, TableSampleUnit,
    Value as SQLValue, ValueWithSpan, VisitMut, VisitorMut,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::keywords::Keyword;
use sqlparser::parser::Parser;
use sqlparser::tokenizer::{Token, TokenWithSpan, Whitespace, Word};

use crate::dyn_visit;
use crate::time_travel::is_keyword;

/// The `USING SAMPLE` clauses taken out of a query, by the position (line, column) of the
/// `SELECT` they belong to. Each must be attached to a statement, or refused.
#[must_use]
#[derive(Default)]
pub(crate) struct QuerySamples(Vec<SampleAt>);

/// A sample clause and the position (line, column) of its `SELECT`.
type SampleAt = ((u64, u64), Box<TableSample>);

impl QuerySamples {
    /// Put each sample on the one table of the FROM clause of its SELECT.
    pub(crate) fn attach<T: VisitMut>(mut self, nodes: &mut [T]) -> PolarsResult<()> {
        if self.0.is_empty() {
            return Ok(());
        }
        let mut attach = Attach(&mut self.0);
        for node in nodes {
            if let ControlFlow::Break(err) = dyn_visit::visit_mut(node, &mut attach) {
                return Err(err);
            }
        }
        self.refuse()
    }

    /// An error if a clause was taken out where no SELECT can take it.
    pub(crate) fn refuse(self) -> PolarsResult<()> {
        polars_ensure!(
            self.0.is_empty(),
            SQLInterface: "`USING SAMPLE` must follow the FROM clause of a SELECT"
        );
        Ok(())
    }
}

struct Attach<'a>(&'a mut Vec<SampleAt>);

impl Attach<'_> {
    fn attach_set_expr(&mut self, body: &mut SetExpr) -> PolarsResult<()> {
        match body {
            SetExpr::Select(select) => {
                let start = select.select_token.0.span.start;
                let Some(i) = self
                    .0
                    .iter()
                    .position(|(at, _)| *at == (start.line, start.column))
                else {
                    return Ok(());
                };
                let (_, sample) = self.0.swap_remove(i);
                let relation = match select.from.as_mut_slice() {
                    [from] if from.joins.is_empty() => &mut from.relation,
                    _ => polars_bail!(
                        SQLInterface: "`USING SAMPLE` is only supported with one table in FROM; put `TABLESAMPLE` after a table instead"
                    ),
                };
                let (TableFactor::Table { sample: slot, .. }
                | TableFactor::Derived { sample: slot, .. }) = relation
                else {
                    polars_bail!(SQLInterface: "`USING SAMPLE` is not supported on {}", relation)
                };
                polars_ensure!(
                    slot.is_none(),
                    SQLInterface: "a table takes one sample clause"
                );
                *slot = Some(TableSampleKind::AfterTableAlias(sample));
                Ok(())
            },
            SetExpr::SetOperation { left, right, .. } => {
                self.attach_set_expr(left)?;
                self.attach_set_expr(right)
            },
            _ => Ok(()),
        }
    }
}

impl VisitorMut for Attach<'_> {
    type Break = PolarsError;

    fn pre_visit_query(&mut self, query: &mut Query) -> ControlFlow<PolarsError> {
        match self.attach_set_expr(&mut query.body) {
            Ok(()) => ControlFlow::Continue(()),
            Err(err) => ControlFlow::Break(err),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Unit {
    Percent,
    Rows,
}

/// A sample clause after its keyword, as the tokens it ends before.
struct SampleSpec {
    quantity: String,
    unit: Unit,
    seed: Option<String>,
    end: usize,
}

fn next_token(tokens: &[TokenWithSpan], from: usize) -> usize {
    (from..tokens.len())
        .find(|&i| !matches!(tokens[i].token, Token::Whitespace(_)))
        .unwrap_or(tokens.len())
}

fn token_at(tokens: &[TokenWithSpan], i: usize) -> &Token {
    tokens.get(i).map_or(&Token::EOF, |t| &t.token)
}

fn sample_method(token: &Token) -> Option<String> {
    let Token::Word(Word {
        value,
        quote_style: None,
        ..
    }) = token
    else {
        return None;
    };
    let method = value.to_lowercase();
    ["bernoulli", "system", "reservoir", "row", "block"]
        .contains(&method.as_str())
        .then_some(method)
}

/// The sample clause starting at token `i` in one of these forms, or `None` if it has none of
/// them:
/// - `[method] (n [PERCENT | % | ROWS]) [REPEATABLE (seed)]`
/// - `n [PERCENT | % | ROWS] [(method [, seed])] [REPEATABLE (seed)]`
/// - `method(n [PERCENT | % | ROWS]) [REPEATABLE (seed)]`
///
/// Without a unit, `n` is a percentage after a method other than `reservoir` in parentheses
/// (`BERNOULLI (10)`), and a number of rows otherwise.
fn parse_sample_spec(tokens: &[TokenWithSpan], i: usize) -> Option<SampleSpec> {
    let mut j = next_token(tokens, i);
    let mut method = None;
    if let Some(m) = sample_method(token_at(tokens, j))
        && matches!(token_at(tokens, next_token(tokens, j + 1)), Token::LParen)
    {
        method = Some(m);
        j = next_token(tokens, j + 1);
    }
    let parenthesized = matches!(token_at(tokens, j), Token::LParen);
    if parenthesized {
        j = next_token(tokens, j + 1);
    }
    let Token::Number(quantity, _) = token_at(tokens, j) else {
        return None;
    };
    let quantity = quantity.clone();
    j = next_token(tokens, j + 1);
    let unit = match token_at(tokens, j) {
        Token::Mod => Some(Unit::Percent),
        t if is_keyword(t, Keyword::PERCENT) => Some(Unit::Percent),
        t if is_keyword(t, Keyword::ROWS) => Some(Unit::Rows),
        _ => None,
    };
    if unit.is_some() {
        j = next_token(tokens, j + 1);
    }
    if parenthesized {
        if !matches!(token_at(tokens, j), Token::RParen) {
            return None;
        }
        j = next_token(tokens, j + 1);
    }
    let mut seed = None;
    let number = |j: usize| match token_at(tokens, j) {
        Token::Number(n, _) => Some(n.clone()),
        _ => None,
    };
    if !parenthesized
        && method.is_none()
        && matches!(token_at(tokens, j), Token::LParen)
        && let Some(m) = sample_method(token_at(tokens, next_token(tokens, j + 1)))
    {
        method = Some(m);
        let mut k = next_token(tokens, next_token(tokens, j + 1) + 1);
        if matches!(token_at(tokens, k), Token::Comma) {
            k = next_token(tokens, k + 1);
            seed = Some(number(k)?);
            k = next_token(tokens, k + 1);
        }
        if !matches!(token_at(tokens, k), Token::RParen) {
            return None;
        }
        j = next_token(tokens, k + 1);
    }
    if is_keyword(token_at(tokens, j), Keyword::REPEATABLE) {
        let open = next_token(tokens, j + 1);
        let value = next_token(tokens, open + 1);
        let close = next_token(tokens, value + 1);
        if !matches!(token_at(tokens, open), Token::LParen)
            || !matches!(token_at(tokens, close), Token::RParen)
        {
            return None;
        }
        seed = Some(number(value)?);
        j = close + 1;
    }
    let unit = unit.unwrap_or(
        if parenthesized && method.as_ref().is_some_and(|m| m != "reservoir") {
            Unit::Percent
        } else {
            Unit::Rows
        },
    );
    Some(SampleSpec {
        quantity,
        unit,
        seed,
        end: j.min(tokens.len()),
    })
}

/// The tokens `(n PERCENT|ROWS) [REPEATABLE (seed)]` of `spec`.
fn spec_tokens(spec: &SampleSpec, span: sqlparser::tokenizer::Span) -> Vec<TokenWithSpan> {
    let unit = match spec.unit {
        Unit::Percent => "PERCENT",
        Unit::Rows => "ROWS",
    };
    let mut tokens = vec![
        Token::Whitespace(Whitespace::Space),
        Token::LParen,
        Token::Number(spec.quantity.clone(), false),
        Token::Whitespace(Whitespace::Space),
        Token::make_keyword(unit),
        Token::RParen,
    ];
    if let Some(seed) = &spec.seed {
        tokens.extend([
            Token::Whitespace(Whitespace::Space),
            Token::make_keyword("REPEATABLE"),
            Token::LParen,
            Token::Number(seed.clone(), false),
            Token::RParen,
        ]);
    }
    tokens.push(Token::Whitespace(Whitespace::Space));
    tokens
        .into_iter()
        .map(|token| TokenWithSpan::new(token, span))
        .collect()
}

/// Rewrite the `TABLESAMPLE` clauses in `tokens` into the form sqlparser parses, and take out
/// the `USING SAMPLE` clauses.
pub(crate) fn rewrite_sample_tokens(tokens: &mut Vec<TokenWithSpan>) -> PolarsResult<QuerySamples> {
    if !tokens.iter().any(|t| {
        is_keyword(&t.token, Keyword::TABLESAMPLE) || is_keyword(&t.token, Keyword::SAMPLE)
    }) {
        return Ok(QuerySamples::default());
    }
    let input = std::mem::take(tokens);
    let mut samples = vec![];
    // The position of the last SELECT, per parenthesis depth.
    let mut selects: Vec<Option<(u64, u64)>> = vec![None];
    let mut i = 0;
    while i < input.len() {
        let token = &input[i].token;
        match token {
            Token::LParen => selects.push(None),
            Token::RParen if selects.len() > 1 => {
                selects.pop();
            },
            t if is_keyword(t, Keyword::SELECT) => {
                let start = input[i].span.start;
                *selects.last_mut().unwrap() = Some((start.line, start.column));
            },
            t if is_keyword(t, Keyword::TABLESAMPLE) => {
                if let Some(spec) = parse_sample_spec(&input, i + 1) {
                    tokens.push(input[i].clone());
                    tokens.extend(spec_tokens(&spec, input[i].span));
                    i = spec.end;
                    continue;
                }
            },
            t if is_keyword(t, Keyword::USING)
                && is_keyword(token_at(&input, next_token(&input, i + 1)), Keyword::SAMPLE) =>
            {
                let sample_at = next_token(&input, i + 1);
                let Some(spec) = parse_sample_spec(&input, sample_at + 1) else {
                    polars_bail!(SQLInterface: "invalid `USING SAMPLE` clause; expected a form such as `USING SAMPLE 10%` or `USING SAMPLE 10 PERCENT (bernoulli, 42)`")
                };
                let Some(select) = *selects.last().unwrap() else {
                    polars_bail!(SQLInterface: "`USING SAMPLE` must follow the FROM clause of a SELECT")
                };
                samples.push((select, parse_table_sample(&spec, input[i].span)?));
                tokens.push(TokenWithSpan::new(
                    Token::Whitespace(Whitespace::Space),
                    input[i].span,
                ));
                i = spec.end;
                continue;
            },
            _ => {},
        }
        tokens.push(input[i].clone());
        i += 1;
    }
    Ok(QuerySamples(samples))
}

/// The sample of `spec`, as sqlparser parses it.
fn parse_table_sample(
    spec: &SampleSpec,
    span: sqlparser::tokenizer::Span,
) -> PolarsResult<Box<TableSample>> {
    let mut tokens = vec![
        TokenWithSpan::new(Token::make_word("t", None), span),
        TokenWithSpan::new(Token::Whitespace(Whitespace::Space), span),
        TokenWithSpan::new(Token::make_keyword("TABLESAMPLE"), span),
    ];
    tokens.extend(spec_tokens(spec, span));
    let factor = Parser::new(&GenericDialect)
        .with_tokens_with_locations(tokens)
        .parse_table_factor()
        .map_err(|err| polars_err!(SQLInterface: "{}", err))?;
    match factor {
        TableFactor::Table {
            sample:
                Some(
                    TableSampleKind::AfterTableAlias(sample)
                    | TableSampleKind::BeforeTableAlias(sample),
                ),
            ..
        } => Ok(sample),
        _ => unreachable!(),
    }
}

/// The fraction of the rows that `sample` keeps, and its seed.
pub(crate) fn sample_fraction(sample: &TableSampleKind) -> PolarsResult<(f64, Option<u64>)> {
    let (TableSampleKind::AfterTableAlias(sample) | TableSampleKind::BeforeTableAlias(sample)) =
        sample;
    let unsupported = || polars_err!(SQLInterface: "unsupported sample clause '{}'; only a percentage is supported, as in `TABLESAMPLE (10 PERCENT)`", sample);
    let (Some(quantity), None, None) = (&sample.quantity, &sample.bucket, &sample.offset) else {
        return Err(unsupported());
    };
    match quantity.unit {
        Some(TableSampleUnit::Percent) => {},
        Some(TableSampleUnit::Rows) => polars_bail!(
            SQLInterface: "sampling a number of rows is not supported; use a percentage, as in `TABLESAMPLE (10 PERCENT)`"
        ),
        None => return Err(unsupported()),
    }
    let SQLExpr::Value(ValueWithSpan {
        value: SQLValue::Number(n, _),
        ..
    }) = &quantity.value
    else {
        return Err(unsupported());
    };
    let percent: f64 = n.parse().map_err(|_| unsupported())?;
    polars_ensure!(
        (0.0..=100.0).contains(&percent),
        SQLInterface: "a sample percentage must be between 0 and 100; found {}", n
    );
    let seed = match &sample.seed {
        None => None,
        Some(seed) => match &seed.value.value {
            SQLValue::Number(n, _) => Some(n.parse().map_err(
                |_| polars_err!(SQLInterface: "a sample seed must be a non-negative integer; found {}", n),
            )?),
            v => polars_bail!(SQLInterface: "a sample seed must be a non-negative integer; found {}", v),
        },
    };
    Ok((percent / 100.0, seed))
}

/// The rows of `lf` that a sample keeping `fraction` of them with `seed` keeps. Each row is
/// kept by a hash of its row index.
pub(crate) fn sample_rows(lf: LazyFrame, fraction: f64, seed: Option<u64>) -> LazyFrame {
    if fraction >= 1.0 {
        return lf;
    }
    let seed = seed.unwrap_or_else(polars_core::random::get_global_random_u64);
    let index = PlSmallStr::from_static("__POLARS_SAMPLE_ROW_INDEX");
    // A row is in the sample if the top 53 bits of its hash, as a fraction of 2^53, are less
    // than `fraction`.
    let threshold = ((fraction * (1u64 << 53) as f64).ceil() as u64) << 11;
    lf.with_row_index(index.clone(), None)
        .filter(col(index.clone()).hash(seed).lt(lit(threshold)))
        .drop(polars_plan::dsl::Selector::ByName {
            names: [index].into(),
            strict: true,
        })
}
