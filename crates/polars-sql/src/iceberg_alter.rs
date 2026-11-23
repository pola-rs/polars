//! The `ALTER TABLE` operations of Iceberg tables that sqlparser does not parse yet:
//! `SET | RESET PARTITIONED BY`, `SET | RESET SORTED BY`, and the tag and branch statements
//! (`CREATE [OR REPLACE] TAG | BRANCH`, `REPLACE TAG | BRANCH`, `DROP TAG | BRANCH`).
//!
//! Temporary: remove once sqlparser parses these statements.

use polars_core::prelude::*;
use sqlparser::ast::{Expr, ObjectName, OrderByExpr, Statement};
use sqlparser::keywords::Keyword;
use sqlparser::parser::{Parser, ParserError};
use sqlparser::tokenizer::Token;

use crate::catalog::{CatalogAlter, CatalogRefKind, CatalogRefMode};
use crate::macros::{MacroStatement, parse_macro_statement};
use crate::sql_expr::{sql_parser, to_sql_interface_err};

/// A statement that sqlparser does not parse.
pub(crate) enum OtherStatement {
    Alter(IcebergAlter),
    Macro(MacroStatement),
}

/// A parsed `ALTER TABLE [IF EXISTS] <name> <operation>`, for one of the operations above.
#[derive(Clone, Debug)]
pub(crate) struct IcebergAlter {
    pub(crate) name: ObjectName,
    pub(crate) if_exists: bool,
    pub(crate) operation: IcebergAlterOp,
}

/// The operation, as written.
#[derive(Clone, Debug)]
pub(crate) enum IcebergAlterOp {
    /// The new partition fields; none for `RESET`.
    Partitioning(Vec<Expr>),
    /// The new sort fields; none for `RESET`.
    SortOrder(Vec<OrderByExpr>),
    /// A [`CatalogAlter::SetRef`] or [`CatalogAlter::DropRef`].
    Ref(CatalogAlter),
}

/// What follows the table name, up to the words that make this one of the operations above.
enum Head {
    Layout {
        reset: bool,
        sort: bool,
    },
    SetRef {
        kind: CatalogRefKind,
        mode: CatalogRefMode,
    },
    DropRef {
        kind: CatalogRefKind,
    },
}

/// The statements of `query`, or the statement that sqlparser does not parse.
pub(crate) fn parse_statements_or_alter(
    query: &str,
) -> PolarsResult<Result<Vec<Statement>, OtherStatement>> {
    let (mut parser, taken) = sql_parser(query)?;
    if let Some(alter) = parse_iceberg_alter(&mut parser)? {
        // a time travel clause has no table to read here
        taken.refuse()?;
        return Ok(Err(OtherStatement::Alter(alter)));
    }
    match parse_macro_statement(&mut parser) {
        Ok(None) => {},
        parsed => {
            let parse = |parser: &mut Parser<'static>| {
                Ok(parse_macro_statement(parser)?.into_iter().collect())
            };
            let parsed = parsed.map(|stmt| stmt.into_iter().collect());
            let stmt = taken.attach_to(query, parsed, parse)?.pop();
            let stmt = stmt.ok_or_else(|| polars_err!(SQLInterface: "invalid macro statement"))?;
            return Ok(Err(OtherStatement::Macro(stmt)));
        },
    }
    Ok(Ok(taken.attach(query, parser.parse_statements())?))
}

/// The single statement of `query`, or the statement that sqlparser does not parse.
pub(crate) fn parse_single_statement_or_alter(
    query: &str,
) -> PolarsResult<Result<Statement, OtherStatement>> {
    Ok(match parse_statements_or_alter(query)? {
        Err(other) => Err(other),
        Ok(statements) => {
            let [stmt]: [Statement; 1] = statements.try_into().map_err(|_| {
                polars_err!(SQLInterface: "one (and only one) statement can be parsed at a time")
            })?;
            Ok(stmt)
        },
    })
}

/// The operation at the start of `parser`; if there is none, `parser` stays at the start.
fn parse_iceberg_alter(parser: &mut Parser) -> PolarsResult<Option<IcebergAlter>> {
    // Up to the words that make this an operation of this module. Anything else is parsed as usual.
    let not_ours = || ParserError::ParserError(String::new());
    let head = parser
        .maybe_parse(|p| {
            // as sqlparser, which skips empty statements
            while p.consume_token(&Token::SemiColon) {}
            p.expect_keywords(&[Keyword::ALTER, Keyword::TABLE])?;
            let if_exists = p.parse_keywords(&[Keyword::IF, Keyword::EXISTS]);
            let only = p.parse_keyword(Keyword::ONLY);
            let name = p.parse_object_name(false)?;
            let head =
                if let Some(keyword) = p.parse_one_of_keywords(&[Keyword::SET, Keyword::RESET]) {
                    let sort = if p.parse_keywords(&[Keyword::PARTITIONED, Keyword::BY]) {
                        false
                    } else if p.parse_keywords(&[Keyword::SORTED, Keyword::BY]) {
                        true
                    } else {
                        return Err(not_ours());
                    };
                    Head::Layout {
                        reset: keyword == Keyword::RESET,
                        sort,
                    }
                } else if p.parse_keyword(Keyword::CREATE) {
                    let replace = p.parse_keywords(&[Keyword::OR, Keyword::REPLACE]);
                    let kind = parse_ref_kind(p).ok_or_else(not_ours)?;
                    let mode = if replace {
                        CatalogRefMode::CreateOrReplace
                    } else if p.parse_keywords(&[Keyword::IF, Keyword::NOT, Keyword::EXISTS]) {
                        CatalogRefMode::CreateIfNotExists
                    } else {
                        CatalogRefMode::Create
                    };
                    Head::SetRef { kind, mode }
                } else if p.parse_keyword(Keyword::REPLACE) {
                    let kind = parse_ref_kind(p).ok_or_else(not_ours)?;
                    Head::SetRef {
                        kind,
                        mode: CatalogRefMode::Replace,
                    }
                } else if p.parse_keyword(Keyword::DROP) {
                    let kind = parse_ref_kind(p).ok_or_else(not_ours)?;
                    // `DROP tag` alone drops a column of that name
                    if !starts_ref_name(p) {
                        return Err(not_ours());
                    }
                    Head::DropRef { kind }
                } else {
                    return Err(not_ours());
                };
            Ok((name, if_exists, only, head))
        })
        .map_err(to_sql_interface_err)?;
    let Some((name, if_exists, only, head)) = head else {
        return Ok(None);
    };
    polars_ensure!(!only, SQLInterface: "`ALTER TABLE ONLY` is not supported");
    let operation = match head {
        Head::Layout { reset: true, sort } => match sort {
            false => IcebergAlterOp::Partitioning(Vec::new()),
            true => IcebergAlterOp::SortOrder(Vec::new()),
        },
        Head::Layout {
            reset: false,
            sort: false,
        } => IcebergAlterOp::Partitioning(parse_fields(
            parser,
            "`SET PARTITIONED BY ()` needs a partition field; `RESET PARTITIONED BY` removes the partitioning",
            Parser::parse_expr,
        )?),
        Head::Layout {
            reset: false,
            sort: true,
        } => IcebergAlterOp::SortOrder(parse_fields(
            parser,
            "`SET SORTED BY ()` needs a sort field; `RESET SORTED BY` removes the sort order",
            Parser::parse_order_by_expr,
        )?),
        Head::SetRef { kind, mode } => IcebergAlterOp::Ref(parse_set_ref(parser, kind, mode)?),
        Head::DropRef { kind } => {
            let if_exists = parser.parse_keywords(&[Keyword::IF, Keyword::EXISTS]);
            let name = parse_ref_name(parser, kind)?;
            IcebergAlterOp::Ref(CatalogAlter::DropRef {
                kind,
                name,
                if_exists,
            })
        },
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
    Ok(Some(IcebergAlter {
        name,
        if_exists,
        operation,
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

/// Whether the next token is the unquoted `word`, any case; if so, it is taken.
fn parse_word(parser: &mut Parser, word: &str) -> bool {
    let next = parser.peek_token();
    let found = matches!(
        &next.token,
        Token::Word(w) if w.quote_style.is_none() && w.value.eq_ignore_ascii_case(word)
    );
    if found {
        parser.next_token();
    }
    found
}

fn parse_ref_kind(parser: &mut Parser) -> Option<CatalogRefKind> {
    if parse_word(parser, "TAG") {
        Some(CatalogRefKind::Tag)
    } else if parse_word(parser, "BRANCH") {
        Some(CatalogRefKind::Branch)
    } else {
        None
    }
}

fn ref_kind_name(kind: CatalogRefKind) -> &'static str {
    match kind {
        CatalogRefKind::Tag => "tag",
        CatalogRefKind::Branch => "branch",
    }
}

/// Whether a ref name (or `IF EXISTS`) follows, rather than the end of a column drop.
fn starts_ref_name(parser: &Parser) -> bool {
    match parser.peek_token().token {
        Token::Word(w) => !matches!(w.keyword, Keyword::CASCADE | Keyword::RESTRICT),
        Token::SingleQuotedString(_) => true,
        _ => false,
    }
}

/// The name of a tag or branch: an identifier, bare or quoted, with its case.
fn parse_ref_name(parser: &mut Parser, kind: CatalogRefKind) -> PolarsResult<String> {
    let kind = ref_kind_name(kind);
    polars_ensure!(
        !matches!(parser.peek_token().token, Token::SingleQuotedString(_)),
        SQLInterface: "a {} name is an identifier: write it bare or in double quotes, not in single quotes", kind
    );
    let name = parser
        .parse_identifier()
        .map_err(to_sql_interface_err)?
        .value;
    polars_ensure!(!name.is_empty(), SQLInterface: "a {} name cannot be empty", kind);
    Ok(name)
}

/// The name and options of a statement that points a tag or branch at a snapshot:
/// `<name> [AS OF VERSION <id>] [RETAIN <n> <unit>] [WITH SNAPSHOT RETENTION ...]`.
fn parse_set_ref(
    parser: &mut Parser,
    kind: CatalogRefKind,
    mode: CatalogRefMode,
) -> PolarsResult<CatalogAlter> {
    let name = parse_ref_name(parser, kind)?;
    // the forms of the read clause (`FOR VERSION AS OF`, `VERSION AS OF`, `AT (...)`)
    polars_ensure!(
        !matches!(
            &parser.peek_token().token,
            Token::Word(w) if w.quote_style.is_none() && (
                matches!(w.keyword, Keyword::FOR | Keyword::AT)
                    || w.value.eq_ignore_ascii_case("VERSION")
            )
        ),
        SQLInterface: "the snapshot of a {} is written `AS OF VERSION <snapshot id>`", ref_kind_name(kind)
    );
    let snapshot_id = if parser.parse_keywords(&[Keyword::AS, Keyword::OF]) {
        if !parse_word(parser, "VERSION") {
            return parser
                .expected("VERSION", parser.peek_token())
                .map_err(to_sql_interface_err);
        }
        Some(parse_snapshot_id(parser)?)
    } else {
        None
    };
    let max_ref_age_ms = match parse_word(parser, "RETAIN") {
        true => Some(parse_duration_ms(parser, "RETAIN")?),
        false => None,
    };
    let (mut min_snapshots_to_keep, mut max_snapshot_age_ms) = (None, None);
    if parser.parse_keyword(Keyword::WITH) {
        if !(parse_word(parser, "SNAPSHOT") && parse_word(parser, "RETENTION")) {
            return parser
                .expected("SNAPSHOT RETENTION", parser.peek_token())
                .map_err(to_sql_interface_err);
        }
        polars_ensure!(
            kind == CatalogRefKind::Branch,
            SQLInterface: "`WITH SNAPSHOT RETENTION` is for branches: a tag points at one snapshot"
        );
        let what = "WITH SNAPSHOT RETENTION";
        let n = parse_positive(parser, what)?;
        if parse_word(parser, "SNAPSHOTS") {
            min_snapshots_to_keep = Some(i32::try_from(n).map_err(
                |_| polars_err!(SQLInterface: "`{}` takes at most {} snapshots", what, i32::MAX),
            )?);
            if matches!(parser.peek_token().token, Token::Number(..)) {
                max_snapshot_age_ms = Some(parse_duration_ms(parser, what)?);
            }
        } else {
            max_snapshot_age_ms = Some(times_unit(parser, n, what)?);
        }
    }
    Ok(CatalogAlter::SetRef {
        kind,
        name,
        mode,
        snapshot_id,
        max_ref_age_ms,
        min_snapshots_to_keep,
        max_snapshot_age_ms,
    })
}

/// The snapshot id of `AS OF VERSION`.
fn parse_snapshot_id(parser: &mut Parser) -> PolarsResult<i64> {
    let negative = parser.consume_token(&Token::Minus);
    let next = parser.next_token();
    match &next.token {
        Token::Number(digits, _) => {
            let digits = if negative {
                format!("-{digits}")
            } else {
                digits.clone()
            };
            digits.parse().map_err(
                |_| polars_err!(SQLInterface: "`AS OF VERSION` takes a snapshot id; found {}", digits),
            )
        },
        Token::SingleQuotedString(_) if !negative => polars_bail!(
            SQLInterface: "`AS OF VERSION` takes a snapshot id, not a tag or branch: the `$refs` table (e.g. `SELECT * FROM t$refs`) lists the snapshot of each"
        ),
        _ => parser
            .expected("a snapshot id", next)
            .map_err(to_sql_interface_err),
    }
}

/// A whole number greater than zero, for the option `what`.
fn parse_positive(parser: &mut Parser, what: &str) -> PolarsResult<i64> {
    let next = parser.next_token();
    match &next.token {
        Token::Number(digits, _) => match digits.parse::<i64>() {
            Ok(n) if n > 0 => Ok(n),
            _ => {
                polars_bail!(SQLInterface: "`{}` takes a whole number greater than 0; found {}", what, digits)
            },
        },
        _ => parser
            .expected("a whole number", next)
            .map_err(to_sql_interface_err),
    }
}

/// `<n> DAYS | HOURS | MINUTES`, in milliseconds.
fn parse_duration_ms(parser: &mut Parser, what: &str) -> PolarsResult<i64> {
    let n = parse_positive(parser, what)?;
    times_unit(parser, n, what)
}

/// `n` times the unit that follows, in milliseconds.
fn times_unit(parser: &mut Parser, n: i64, what: &str) -> PolarsResult<i64> {
    let unit_ms: i64 = if parse_word(parser, "DAYS") {
        86_400_000
    } else if parse_word(parser, "HOURS") {
        3_600_000
    } else if parse_word(parser, "MINUTES") {
        60_000
    } else {
        return parser
            .expected("DAYS, HOURS or MINUTES", parser.peek_token())
            .map_err(to_sql_interface_err);
    };
    n.checked_mul(unit_ms)
        .ok_or_else(|| polars_err!(SQLInterface: "`{}` is too long", what))
}
