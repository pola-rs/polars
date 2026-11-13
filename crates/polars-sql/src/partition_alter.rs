//! `ALTER TABLE ... SET PARTITIONED BY (...)` and `ALTER TABLE ... RESET PARTITIONED BY`, which
//! sqlparser does not parse yet.
//!
//! Temporary: remove once sqlparser parses these statements.

use polars_core::prelude::*;
use sqlparser::ast::{Expr, ObjectName, Statement};
use sqlparser::keywords::Keyword;
use sqlparser::parser::{Parser, ParserError};
use sqlparser::tokenizer::Token;

use crate::sql_expr::{sql_parser, to_sql_interface_err};

/// A parsed `ALTER TABLE [IF EXISTS] <name> SET PARTITIONED BY (<fields>)` or
/// `ALTER TABLE [IF EXISTS] <name> RESET PARTITIONED BY`.
#[derive(Clone, Debug)]
pub(crate) struct PartitionAlter {
    pub(crate) name: ObjectName,
    pub(crate) if_exists: bool,
    /// The partition fields, as written; empty for `RESET PARTITIONED BY`.
    pub(crate) fields: Vec<Expr>,
}

/// The statements of `query`, or its partition change.
pub(crate) fn parse_statements_or_alter(
    query: &str,
) -> PolarsResult<Result<Vec<Statement>, PartitionAlter>> {
    let (mut parser, versions) = sql_parser(query)?;
    if let Some(alter) = parse_partition_alter(&mut parser)? {
        // a time travel clause has no table to read here
        versions.refuse()?;
        return Ok(Err(alter));
    }
    let mut stmts = parser.parse_statements().map_err(to_sql_interface_err)?;
    versions.attach(&mut stmts)?;
    Ok(Ok(stmts))
}

/// The single statement of `query`, or its partition change.
pub(crate) fn parse_single_statement_or_alter(
    query: &str,
) -> PolarsResult<Result<Statement, PartitionAlter>> {
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

/// The partition change at the start of `parser`; if there is none, `parser` stays at the start.
fn parse_partition_alter(parser: &mut Parser) -> PolarsResult<Option<PartitionAlter>> {
    // Up to the keywords that make this a partition change. Anything else is parsed as usual.
    let head = parser
        .maybe_parse(|p| {
            // as sqlparser, which skips empty statements
            while p.consume_token(&Token::SemiColon) {}
            p.expect_keywords(&[Keyword::ALTER, Keyword::TABLE])?;
            let if_exists = p.parse_keywords(&[Keyword::IF, Keyword::EXISTS]);
            let only = p.parse_keyword(Keyword::ONLY);
            let name = p.parse_object_name(false)?;
            let reset = if p.parse_keywords(&[Keyword::SET, Keyword::PARTITIONED, Keyword::BY]) {
                false
            } else if p.parse_keywords(&[Keyword::RESET, Keyword::PARTITIONED, Keyword::BY]) {
                true
            } else {
                return Err(ParserError::ParserError(String::new()));
            };
            Ok((name, if_exists, only, reset))
        })
        .map_err(to_sql_interface_err)?;
    let Some((name, if_exists, only, reset)) = head else {
        return Ok(None);
    };
    polars_ensure!(!only, SQLInterface: "`ALTER TABLE ONLY` is not supported");
    let fields = if reset {
        Vec::new()
    } else {
        parser
            .expect_token(&Token::LParen)
            .map_err(to_sql_interface_err)?;
        polars_ensure!(
            !parser.consume_token(&Token::RParen),
            SQLInterface: "`SET PARTITIONED BY ()` needs a partition field; `RESET PARTITIONED BY` removes the partitioning"
        );
        let fields = parser
            .parse_comma_separated(Parser::parse_expr)
            .map_err(to_sql_interface_err)?;
        parser
            .expect_token(&Token::RParen)
            .map_err(to_sql_interface_err)?;
        fields
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
    Ok(Some(PartitionAlter {
        name,
        if_exists,
        fields,
    }))
}
