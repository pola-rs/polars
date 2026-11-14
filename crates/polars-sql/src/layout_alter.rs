//! `ALTER TABLE ... SET | RESET PARTITIONED BY` and `ALTER TABLE ... SET | RESET SORTED BY`, which
//! sqlparser does not parse yet.
//!
//! Temporary: remove once sqlparser parses these statements.

use polars_core::prelude::*;
use sqlparser::ast::{Expr, ObjectName, OrderByExpr, Statement};
use sqlparser::keywords::Keyword;
use sqlparser::parser::{Parser, ParserError};
use sqlparser::tokenizer::Token;

use crate::sql_expr::{sql_parser, to_sql_interface_err};

/// A parsed `ALTER TABLE [IF EXISTS] <name> SET | RESET PARTITIONED BY ...` or
/// `ALTER TABLE [IF EXISTS] <name> SET | RESET SORTED BY ...`.
#[derive(Clone, Debug)]
pub(crate) struct LayoutAlter {
    pub(crate) name: ObjectName,
    pub(crate) if_exists: bool,
    pub(crate) layout: Layout,
}

/// The new layout, as written; no fields for `RESET`.
#[derive(Clone, Debug)]
pub(crate) enum Layout {
    Partitioning(Vec<Expr>),
    SortOrder(Vec<OrderByExpr>),
}

/// The statements of `query`, or its layout change.
pub(crate) fn parse_statements_or_alter(
    query: &str,
) -> PolarsResult<Result<Vec<Statement>, LayoutAlter>> {
    let (mut parser, versions) = sql_parser(query)?;
    if let Some(alter) = parse_layout_alter(&mut parser)? {
        // a time travel clause has no table to read here
        versions.refuse()?;
        return Ok(Err(alter));
    }
    let mut stmts = parser.parse_statements().map_err(to_sql_interface_err)?;
    versions.attach(&mut stmts)?;
    Ok(Ok(stmts))
}

/// The single statement of `query`, or its layout change.
pub(crate) fn parse_single_statement_or_alter(
    query: &str,
) -> PolarsResult<Result<Statement, LayoutAlter>> {
    Ok(match parse_statements_or_alter(query)? {
        Err(alter) => Err(alter),
        Ok(statements) => {
            let [stmt]: [Statement; 1] = statements.try_into().map_err(|_| {
                polars_err!(SQLInterface: "one (and only one) statement can be parsed at a time")
            })?;
            Ok(stmt)
        },
    })
}

/// The layout change at the start of `parser`; if there is none, `parser` stays at the start.
fn parse_layout_alter(parser: &mut Parser) -> PolarsResult<Option<LayoutAlter>> {
    // Up to the keywords that make this a layout change. Anything else is parsed as usual.
    let head = parser
        .maybe_parse(|p| {
            // as sqlparser, which skips empty statements
            while p.consume_token(&Token::SemiColon) {}
            p.expect_keywords(&[Keyword::ALTER, Keyword::TABLE])?;
            let if_exists = p.parse_keywords(&[Keyword::IF, Keyword::EXISTS]);
            let only = p.parse_keyword(Keyword::ONLY);
            let name = p.parse_object_name(false)?;
            let reset = match p.parse_one_of_keywords(&[Keyword::SET, Keyword::RESET]) {
                Some(Keyword::SET) => false,
                Some(_) => true,
                None => return Err(ParserError::ParserError(String::new())),
            };
            let sort = if p.parse_keywords(&[Keyword::PARTITIONED, Keyword::BY]) {
                false
            } else if p.parse_keywords(&[Keyword::SORTED, Keyword::BY]) {
                true
            } else {
                return Err(ParserError::ParserError(String::new()));
            };
            Ok((name, if_exists, only, reset, sort))
        })
        .map_err(to_sql_interface_err)?;
    let Some((name, if_exists, only, reset, sort)) = head else {
        return Ok(None);
    };
    polars_ensure!(!only, SQLInterface: "`ALTER TABLE ONLY` is not supported");
    let layout = match (reset, sort) {
        (true, false) => Layout::Partitioning(Vec::new()),
        (true, true) => Layout::SortOrder(Vec::new()),
        (false, false) => Layout::Partitioning(parse_fields(
            parser,
            "`SET PARTITIONED BY ()` needs a partition field; `RESET PARTITIONED BY` removes the partitioning",
            Parser::parse_expr,
        )?),
        (false, true) => Layout::SortOrder(parse_fields(
            parser,
            "`SET SORTED BY ()` needs a sort field; `RESET SORTED BY` removes the sort order",
            Parser::parse_order_by_expr,
        )?),
    };
    polars_ensure!(
        !parser.consume_token(&Token::Comma),
        SQLInterface: "`ALTER TABLE` takes one operation per statement"
    );
    let ended = parser.consume_token(&Token::SemiColon);
    while parser.consume_token(&Token::SemiColon) {}
    let next = parser.peek_token();
    if next.token != Token::EOF {
        polars_ensure!(
            !ended,
            SQLInterface: "one (and only one) statement can be parsed at a time"
        );
        return parser
            .expected("end of statement", next)
            .map_err(to_sql_interface_err);
    }
    Ok(Some(LayoutAlter {
        name,
        if_exists,
        layout,
    }))
}

/// The parenthesized, non-empty fields at `parser`; `empty` is the message for `()`.
fn parse_fields<'a, T>(
    parser: &mut Parser<'a>,
    empty: &str,
    field: impl FnMut(&mut Parser<'a>) -> Result<T, ParserError>,
) -> PolarsResult<Vec<T>> {
    parser
        .expect_token(&Token::LParen)
        .map_err(to_sql_interface_err)?;
    polars_ensure!(!parser.consume_token(&Token::RParen), SQLInterface: "{}", empty);
    let fields = parser
        .parse_comma_separated(field)
        .map_err(to_sql_interface_err)?;
    parser
        .expect_token(&Token::RParen)
        .map_err(to_sql_interface_err)?;
    Ok(fields)
}
