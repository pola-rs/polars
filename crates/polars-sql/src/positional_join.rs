//! `POSITIONAL JOIN`, which sqlparser does not parse. [`extract_positional_joins`] takes the
//! word `POSITIONAL` out before a `JOIN`, and [`PositionalJoins::attach`] marks the joins it
//! was taken from, found by position, with [`positional_join`].
use std::ops::ControlFlow;

use polars_core::prelude::*;
use sqlparser::ast::{
    JoinConstraint, JoinOperator, Select, Spanned, Statement, TableFactor, TableWithJoins, Visit,
    Visitor, VisitorMut,
};
use sqlparser::keywords::Keyword;
use sqlparser::tokenizer::{Location, Token, TokenWithSpan};

use crate::dyn_visit;

/// The positions of the `POSITIONAL` words taken out of a query. Each must be attached to a
/// join, or refused.
#[must_use]
pub(crate) struct PositionalJoins(Vec<Location>);

/// The operator of a positional join. The parser never gives a `CROSS JOIN` a `NATURAL`
/// constraint.
pub(crate) fn positional_join() -> JoinOperator {
    JoinOperator::CrossJoin(JoinConstraint::Natural)
}

pub(crate) fn is_positional_join(op: &JoinOperator) -> bool {
    matches!(op, JoinOperator::CrossJoin(JoinConstraint::Natural))
}

/// Whether `node` holds a positional join.
pub(crate) fn has_positional_join(node: &(impl Visit + ?Sized)) -> bool {
    struct Find;

    impl Visitor for Find {
        type Break = ();

        fn pre_visit_select(&mut self, select: &Select) -> ControlFlow<()> {
            find_in(&select.from)
        }

        fn pre_visit_table_factor(&mut self, factor: &TableFactor) -> ControlFlow<()> {
            match factor {
                TableFactor::NestedJoin {
                    table_with_joins, ..
                } => find_in(std::slice::from_ref(table_with_joins)),
                _ => ControlFlow::Continue(()),
            }
        }
    }

    fn find_in(items: &[TableWithJoins]) -> ControlFlow<()> {
        match items
            .iter()
            .flat_map(|item| &item.joins)
            .any(|join| is_positional_join(&join.join_operator))
        {
            true => ControlFlow::Break(()),
            false => ControlFlow::Continue(()),
        }
    }

    dyn_visit::visit(node, &mut Find).is_break()
}

impl PositionalJoins {
    /// Mark each plain `JOIN` without a condition that a `POSITIONAL` word was taken from.
    pub(crate) fn attach(mut self, stmts: &mut [Statement]) -> PolarsResult<()> {
        if self.0.is_empty() {
            return Ok(());
        }
        for stmt in stmts {
            let _ = dyn_visit::visit_mut(stmt, &mut self);
        }
        self.refuse()
    }

    /// An error if a `POSITIONAL` word was taken out where no join can take it.
    pub(crate) fn refuse(self) -> PolarsResult<()> {
        polars_ensure!(
            self.0.is_empty(),
            SQLInterface: "POSITIONAL JOIN does not take a join condition"
        );
        Ok(())
    }

    fn mark(&mut self, item: &mut TableWithJoins) {
        let mut before = item.relation.span().end;
        for join in &mut item.joins {
            let after = join.relation.span().start;
            if join.join_operator == JoinOperator::Join(JoinConstraint::None)
                && let Some(i) = self.0.iter().position(|at| before < *at && *at < after)
            {
                self.0.swap_remove(i);
                join.join_operator = positional_join();
            }
            before = join.span().end;
        }
    }
}

impl VisitorMut for PositionalJoins {
    type Break = ();

    fn pre_visit_select(&mut self, select: &mut Select) -> ControlFlow<()> {
        for item in &mut select.from {
            self.mark(item);
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<()> {
        if let TableFactor::NestedJoin {
            table_with_joins, ..
        } = factor
        {
            self.mark(table_with_joins);
        }
        ControlFlow::Continue(())
    }
}

/// Take each unquoted `POSITIONAL` that comes right before `JOIN` out of `tokens`, unless it
/// stands where a name goes.
pub(crate) fn extract_positional_joins(tokens: &mut Vec<TokenWithSpan>) -> PositionalJoins {
    let mut found = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let is_positional = matches!(
            &tokens[i].token,
            Token::Word(w) if w.quote_style.is_none() && w.value.eq_ignore_ascii_case("POSITIONAL")
        );
        if is_positional
            && tokens[i + 1..]
                .iter()
                .find(|t| significant(t))
                .is_some_and(|t| is_keyword(t, Keyword::JOIN))
            && !name_follows(&tokens[..i])
        {
            found.push(tokens.remove(i).span.start);
        } else {
            i += 1;
        }
    }
    PositionalJoins(found)
}

fn significant(token: &TokenWithSpan) -> bool {
    !matches!(token.token, Token::Whitespace(_))
}

fn is_keyword(token: &TokenWithSpan, keyword: Keyword) -> bool {
    matches!(&token.token, Token::Word(w) if w.keyword == keyword && w.quote_style.is_none())
}

/// Whether a name goes right after `tokens`: after `.`, `,`, `(`, or a keyword such as `FROM`
/// or `AS` that is not itself a name.
fn name_follows(tokens: &[TokenWithSpan]) -> bool {
    let mut before = tokens.iter().rev().filter(|t| significant(t));
    match before.next().map(|t| &t.token) {
        Some(Token::Period | Token::Comma | Token::LParen) => true,
        Some(Token::Word(w)) => {
            w.quote_style.is_none()
                && matches!(
                    w.keyword,
                    Keyword::AS
                        | Keyword::FROM
                        | Keyword::JOIN
                        | Keyword::LATERAL
                        | Keyword::TABLE
                        | Keyword::INTO
                        | Keyword::UPDATE
                        | Keyword::USING
                )
                && !before
                    .next()
                    .is_some_and(|t| t.token == Token::Period || is_keyword(t, Keyword::AS))
        },
        _ => false,
    }
}
