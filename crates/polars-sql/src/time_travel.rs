//! Time travel on catalog tables: `[FOR] VERSION AS OF x`, `[FOR] TIMESTAMP AS OF ts`,
//! `FOR SYSTEM_TIME AS OF ts` and `AT (VERSION|TIMESTAMP => x)`.
//!
//! sqlparser's generic dialect parses none of them, so [`extract_table_versions`] takes the
//! clause out of the tokens and [`TableVersions::attach`] puts it on the table factor it
//! follows. A clause that no table factor claims is an error, so it is never lost.
//! Temporary: replace this pass once a sqlparser release parses these forms.
use std::ops::ControlFlow;

use polars_core::prelude::*;
use polars_utils::aliases::PlHashMap;
use sqlparser::ast::{
    DataType as SQLDataType, Expr as SQLExpr, ObjectName, Statement, TableFactor, TableVersion,
    TimezoneInfo, UnaryOperator, Value as SQLValue, VisitorMut,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::keywords::{Keyword, RESERVED_FOR_TABLE_ALIAS};
use sqlparser::parser::Parser;
use sqlparser::tokenizer::{Token, TokenWithSpan};

use crate::types::{is_iso_date, is_iso_datetime, strip_utc_offset};
use crate::{SQLContext, dyn_visit};

/// The time travel clauses taken out of a query, by the position (line, column) of the table
/// name each follows. Each must be attached to a statement, or refused.
#[must_use]
pub(crate) struct TableVersions(PlHashMap<(u64, u64), TableVersion>);

impl TableVersions {
    /// Put each clause on the table factor whose name starts at its position.
    pub(crate) fn attach(mut self, stmts: &mut [Statement]) -> PolarsResult<()> {
        if self.0.is_empty() {
            return Ok(());
        }
        let mut attach = Attach(&mut self.0);
        for stmt in stmts {
            if let ControlFlow::Break(err) = dyn_visit::visit_mut(stmt, &mut attach) {
                return Err(err);
            }
        }
        self.refuse()
    }

    /// An error if a clause was taken out where no table factor can take it.
    pub(crate) fn refuse(self) -> PolarsResult<()> {
        match self.0.iter().min_by_key(|(at, _)| **at) {
            None => Ok(()),
            Some((_, version)) => polars_bail!(
                SQLInterface: "`{}` must follow the name of a table", version
            ),
        }
    }
}

struct Attach<'a>(&'a mut PlHashMap<(u64, u64), TableVersion>);

impl VisitorMut for Attach<'_> {
    type Break = PolarsError;

    fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<PolarsError> {
        let TableFactor::Table { name, version, .. } = factor else {
            return ControlFlow::Continue(());
        };
        if let Some(found) = name_start(name).and_then(|at| self.0.remove(&at)) {
            if version.is_some() {
                return ControlFlow::Break(polars_err!(
                    SQLInterface: "table '{}' takes one time travel clause", name
                ));
            }
            *version = Some(found);
        }
        ControlFlow::Continue(())
    }
}

fn name_start(name: &ObjectName) -> Option<(u64, u64)> {
    let start = name.0.first()?.as_ident()?.span.start;
    Some((start.line, start.column))
}

/// Take the time travel clauses out of `tokens` (see the module docs), in one pass.
pub(crate) fn extract_table_versions(
    tokens: &mut Vec<TokenWithSpan>,
) -> PolarsResult<TableVersions> {
    let input = std::mem::take(tokens);
    let mut found = PlHashMap::new();
    // Per output token: whether a comma or `(` starts a FROM item, and for a `)` the index of
    // its `(`.
    let mut starts_item = Vec::with_capacity(input.len());
    let mut opened_at = Vec::with_capacity(input.len());
    let mut open = Vec::new();
    // Whether the tokens are in a FROM list, per parenthesis depth.
    let mut in_from = vec![false];
    let mut i = 0;
    while i < input.len() {
        if let Some((kind, at_form, value_start)) = clause_head(&input, i)
            && let Some(name) = table_name_before(tokens, &starts_item, &opened_at)
        {
            let (value, end) = clause_value(&input, value_start, at_form)?;
            let version = match kind {
                Kind::Version => TableVersion::VersionAsOf(value),
                Kind::Timestamp => TableVersion::TimestampAsOf(value),
            };
            let at = tokens[name].span.start;
            found.insert((at.line, at.column), version);
            i = end;
            continue;
        }
        let token = &input[i].token;
        let from = in_from.last().copied().unwrap_or(false);
        starts_item.push(from && matches!(token, Token::Comma | Token::LParen));
        opened_at.push(usize::MAX);
        match token {
            Token::LParen => {
                open.push(tokens.len());
                in_from.push(from);
            },
            Token::RParen => {
                if let Some(at) = open.pop() {
                    *opened_at.last_mut().unwrap() = at;
                }
                if in_from.len() > 1 {
                    in_from.pop();
                }
            },
            // a keyword after `AS` is an alias
            Token::Word(w)
                if w.quote_style.is_none()
                    && !prev(tokens, tokens.len())
                        .is_some_and(|j| is_keyword(&tokens[j].token, Keyword::AS)) =>
            {
                use Keyword::*;
                // clauses that end a FROM list, and can't appear inside an expression
                let ends = [
                    SELECT, WHERE, GROUP, HAVING, ORDER, LIMIT, OFFSET, UNION, EXCEPT, INTERSECT,
                    QUALIFY, WINDOW, RETURNING,
                ];
                if let Some(state) = in_from.last_mut() {
                    if matches!(w.keyword, FROM | JOIN | USING) {
                        *state = true;
                    } else if ends.contains(&w.keyword) {
                        *state = false;
                    }
                }
            },
            _ => {},
        }
        tokens.push(input[i].clone());
        i += 1;
    }
    Ok(TableVersions(found))
}

/// The kind of value a time travel clause selects by.
#[derive(Clone, Copy)]
enum Kind {
    Version,
    Timestamp,
}

fn is_keyword(token: &Token, keyword: Keyword) -> bool {
    matches!(token, Token::Word(w) if w.keyword == keyword && w.quote_style.is_none())
}

fn is_significant(token: &Token) -> bool {
    !matches!(token, Token::Whitespace(_))
}

/// The index of the significant token after `i`.
fn next(tokens: &[TokenWithSpan], i: usize) -> Option<usize> {
    (i + 1..tokens.len()).find(|&j| is_significant(&tokens[j].token))
}

/// The index of the significant token before `i`.
fn prev(tokens: &[TokenWithSpan], i: usize) -> Option<usize> {
    (0..i).rev().find(|&j| is_significant(&tokens[j].token))
}

/// Whether the significant tokens from `i` are `keywords`; the index after the last.
fn keywords_at(tokens: &[TokenWithSpan], i: usize, keywords: &[Keyword]) -> Option<usize> {
    let mut at = Some(i);
    let mut last = i;
    for keyword in keywords {
        let j = at?;
        if !is_keyword(&tokens[j].token, *keyword) {
            return None;
        }
        last = j;
        at = next(tokens, j);
    }
    Some(last + 1)
}

/// A clause head at `i`: its kind, whether it is the `AT (... => ...)` form, and where its
/// value starts.
fn clause_head(tokens: &[TokenWithSpan], i: usize) -> Option<(Kind, bool, usize)> {
    use Keyword::{AS, AT, FOR, OF, SYSTEM_TIME, TIMESTAMP, VERSION};
    for (kind, keyword) in [(Kind::Version, VERSION), (Kind::Timestamp, TIMESTAMP)] {
        if let Some(end) = keywords_at(tokens, i, &[FOR, keyword, AS, OF])
            .or_else(|| keywords_at(tokens, i, &[keyword, AS, OF]))
        {
            return Some((kind, false, end));
        }
    }
    if let Some(end) = keywords_at(tokens, i, &[FOR, SYSTEM_TIME, AS, OF]) {
        return Some((Kind::Timestamp, false, end));
    }
    if is_keyword(&tokens[i].token, AT) {
        let open = next(tokens, i)?;
        let word = next(tokens, open)?;
        let arrow = next(tokens, word)?;
        if tokens[open].token == Token::LParen && tokens[arrow].token == Token::RArrow {
            let kind = match &tokens[word].token {
                t if is_keyword(t, VERSION) => Kind::Version,
                t if is_keyword(t, TIMESTAMP) => Kind::Timestamp,
                _ => return None,
            };
            return Some((kind, true, arrow + 1));
        }
    }
    None
}

/// A word that can be part of a table name or be its alias without `AS`.
fn is_name_word(token: &Token) -> bool {
    match token {
        Token::Word(w) if w.quote_style.is_some() => true,
        Token::Word(w) => {
            !RESERVED_FOR_TABLE_ALIAS.contains(&w.keyword)
                && !matches!(w.keyword, Keyword::FROM | Keyword::AS)
        },
        _ => false,
    }
}

/// The first token of the dotted name ending at `end`, if a FROM item starts there: after
/// `FROM`, `JOIN` or `USING`, or a comma or the parenthesis of a grouped relation in a FROM
/// list (`starts_item`). A part next to a `.` may be any word.
fn table_name_ending_at(
    tokens: &[TokenWithSpan],
    starts_item: &[bool],
    end: usize,
) -> Option<usize> {
    let is_dot = |j: Option<usize>| j.is_some_and(|j| tokens[j].token == Token::Period);
    let mut start = end;
    let qualified = is_dot(prev(tokens, end));
    let is_part = |j: usize, qualified: bool| match &tokens[j].token {
        Token::Word(_) if qualified => true,
        token => is_name_word(token),
    };
    if !is_part(end, qualified) {
        return None;
    }
    while let Some(dot) = prev(tokens, start)
        && tokens[dot].token == Token::Period
    {
        let part = prev(tokens, dot)?;
        if !is_part(part, true) {
            return None;
        }
        start = part;
    }
    let before = prev(tokens, start)?;
    let starts = starts_item[before]
        || [Keyword::FROM, Keyword::JOIN, Keyword::USING]
            .iter()
            .any(|k| is_keyword(&tokens[before].token, *k));
    starts.then_some(start)
}

/// The first token of the table name that a clause after the last of `tokens` follows: right
/// after the name, or after its `[AS] alias [(columns)]`.
fn table_name_before(
    tokens: &[TokenWithSpan],
    starts_item: &[bool],
    opened_at: &[usize],
) -> Option<usize> {
    let last = prev(tokens, tokens.len())?;
    if let Some(name) = table_name_ending_at(tokens, starts_item, last) {
        return Some(name);
    }
    let mut alias = last;
    if tokens[alias].token == Token::RParen {
        let open = *opened_at.get(alias).filter(|&&at| at != usize::MAX)?;
        alias = prev(tokens, open)?;
    }
    let mut end = prev(tokens, alias)?;
    // after an explicit `AS`, any word is an alias
    if is_keyword(&tokens[end].token, Keyword::AS) {
        if !matches!(tokens[alias].token, Token::Word(_)) {
            return None;
        }
        end = prev(tokens, end)?;
    } else if !is_name_word(&tokens[alias].token) {
        return None;
    }
    table_name_ending_at(tokens, starts_item, end)
}

/// The literal value of a clause starting at `start`, and the index after the clause.
fn clause_value(
    tokens: &[TokenWithSpan],
    start: usize,
    at_form: bool,
) -> PolarsResult<(SQLExpr, usize)> {
    // a literal is a few tokens: parse from a slice of 16 significant ones, so the work stays
    // linear, and only a prefix expression, so a following alias is not taken as an operator
    let stop = (start..tokens.len())
        .filter(|&j| is_significant(&tokens[j].token))
        .nth(16)
        .unwrap_or(tokens.len());
    let mut parser =
        Parser::new(&GenericDialect).with_tokens_with_locations(tokens[start..stop].to_vec());
    let value = parser
        .parse_prefix()
        .map_err(|e| polars_err!(SQLInterface: "invalid time travel value: {}", e))?;
    let mut end = start + parser.index();
    if at_form {
        let close = next(tokens, end - 1)
            .filter(|&j| tokens[j].token == Token::RParen)
            .ok_or_else(|| polars_err!(SQLInterface: "expected `)` after the time travel value `{}`", value))?;
        end = close + 1;
    }
    polars_ensure!(
        is_literal(&value),
        SQLInterface: "time travel takes a literal value, not `{}`", value
    );
    Ok((value, end))
}

fn is_literal(expr: &SQLExpr) -> bool {
    match expr {
        SQLExpr::Value(v) => matches!(
            v.value,
            SQLValue::Number(..) | SQLValue::SingleQuotedString(_)
        ),
        SQLExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => matches!(&**expr, SQLExpr::Value(v) if matches!(v.value, SQLValue::Number(..))),
        SQLExpr::TypedString(_) => true,
        _ => false,
    }
}

/// What a time travel clause selects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VersionSpec {
    /// A snapshot by its id.
    Snapshot(i64),
    /// A branch or tag by its name.
    Ref(String),
    /// The snapshot current on the main branch at this time, in microseconds since the epoch.
    Timestamp(i64),
}

/// Marks the name part that holds the version of a versioned catalog table. No table name
/// starts with it.
const MARKER: char = '\0';

impl VersionSpec {
    /// The version a clause selects. A timestamp without time zone is UTC.
    pub(crate) fn from_table_version(version: &TableVersion) -> PolarsResult<Self> {
        let number = |expr: &SQLExpr| match expr {
            SQLExpr::Value(v) => match &v.value {
                SQLValue::Number(n, _) => Some(n.to_string()),
                _ => None,
            },
            SQLExpr::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } => match &**expr {
                SQLExpr::Value(v) => match &v.value {
                    SQLValue::Number(n, _) => Some(format!("-{n}")),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        };
        match version {
            TableVersion::VersionAsOf(expr) => {
                if let Some(n) = number(expr) {
                    let id = n.parse::<i64>().map_err(
                        |_| polars_err!(SQLInterface: "invalid snapshot id {}: expected a 64-bit integer", n),
                    )?;
                    return Ok(Self::Snapshot(id));
                }
                match expr {
                    SQLExpr::Value(v) => match &v.value {
                        SQLValue::SingleQuotedString(name) => Ok(Self::Ref(name.clone())),
                        _ => {
                            polars_bail!(SQLInterface: "VERSION AS OF takes a snapshot id or a branch or tag name, not `{}`", expr)
                        },
                    },
                    _ => {
                        polars_bail!(SQLInterface: "VERSION AS OF takes a snapshot id or a branch or tag name, not `{}`", expr)
                    },
                }
            },
            TableVersion::TimestampAsOf(expr) => {
                polars_ensure!(
                    number(expr).is_none(),
                    SQLInterface: "TIMESTAMP AS OF takes a timestamp, not the number {} (for a snapshot id, use VERSION AS OF)", expr
                );
                Ok(Self::Timestamp(timestamp_micros(expr)?))
            },
            other => polars_bail!(SQLInterface: "unsupported time travel clause `{}`", other),
        }
    }

    /// The name part that holds this version in a versioned table name.
    pub(crate) fn marker(&self) -> String {
        match self {
            Self::Snapshot(id) => format!("{MARKER}snapshot:{id}"),
            Self::Ref(name) => format!("{MARKER}ref:{name}"),
            Self::Timestamp(us) => format!("{MARKER}timestamp:{us}"),
        }
    }

    fn from_marker(part: &str) -> Option<Self> {
        let (kind, value) = part.strip_prefix(MARKER)?.split_once(':')?;
        match kind {
            "snapshot" => value.parse().ok().map(Self::Snapshot),
            "ref" => Some(Self::Ref(value.to_owned())),
            "timestamp" => value.parse().ok().map(Self::Timestamp),
            _ => None,
        }
    }
}

impl std::fmt::Display for VersionSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Snapshot(id) => write!(f, "VERSION AS OF {id}"),
            Self::Ref(name) => write!(f, "VERSION AS OF '{}'", name.replace('\'', "''")),
            Self::Timestamp(us) => match chrono::DateTime::from_timestamp_micros(*us) {
                Some(ts) => write!(
                    f,
                    "TIMESTAMP AS OF '{}'",
                    ts.format("%Y-%m-%d %H:%M:%S%.f%:z")
                ),
                None => write!(f, "TIMESTAMP AS OF {us}"),
            },
        }
    }
}

/// The instant of a timestamp literal: ISO text, read as `CAST('<text>' AS TIMESTAMPTZ)`
/// reads it. A UTC offset in the text is kept (a `TIMESTAMP` literal would drop it), no
/// offset is UTC. A `DATE` is its midnight.
fn timestamp_micros(expr: &SQLExpr) -> PolarsResult<i64> {
    let invalid = || polars_err!(SQLInterface: "TIMESTAMP AS OF takes an ISO timestamp or date, not `{}`", expr);
    let (text, date_only) = match expr {
        SQLExpr::Value(v) => match &v.value {
            SQLValue::SingleQuotedString(text) => (text, false),
            _ => return Err(invalid()),
        },
        SQLExpr::TypedString(typed) => match (&typed.data_type, &typed.value.value) {
            (SQLDataType::Date, SQLValue::SingleQuotedString(text)) => (text, true),
            (
                SQLDataType::Timestamp(_, TimezoneInfo::WithTimeZone | TimezoneInfo::Tz),
                SQLValue::SingleQuotedString(text),
            ) => (text, false),
            (
                SQLDataType::Timestamp(..) | SQLDataType::Datetime(_),
                SQLValue::SingleQuotedString(text),
            ) => {
                // Polars SQL drops such an offset elsewhere, other engines keep it
                polars_ensure!(
                    strip_utc_offset(text) == text,
                    SQLInterface: "TIMESTAMP AS OF: `{}` has a UTC offset but no time zone type; write TIMESTAMPTZ '{}' or '{}'", expr, text, text
                );
                (text, false)
            },
            _ => return Err(invalid()),
        },
        _ => return Err(invalid()),
    };
    let iso = is_iso_date(text) || (!date_only && is_iso_datetime(text));
    polars_ensure!(iso, SQLInterface: "{}", invalid());
    let value = SQLValue::SingleQuotedString(text.clone());
    let df = SQLContext::new()
        .execute(&format!("SELECT CAST({value} AS TIMESTAMPTZ) AS ts"))
        .and_then(|lf| lf.collect())
        .map_err(|_| invalid())?;
    let ts = df
        .column("ts")?
        .cast(&DataType::Datetime(TimeUnit::Microseconds, None))?;
    ts.cast(&DataType::Int64)?.i64()?.get(0).ok_or_else(invalid)
}

/// Split a table key into the table name and its version, if versioned.
pub(crate) fn split_version(key: &[String]) -> (&[String], Option<VersionSpec>) {
    match key.split_last() {
        Some((last, name)) => match VersionSpec::from_marker(last) {
            Some(spec) => (name, Some(spec)),
            None => (key, None),
        },
        None => (key, None),
    }
}

/// Whether a name part is reserved for versions; no written name may have one.
pub(crate) fn is_version_part(part: &str) -> bool {
    part.starts_with(MARKER)
}

/// A dotted table key without its version, if versioned.
pub(crate) fn unversioned_key(key: &str) -> &str {
    key.find(&format!(".{MARKER}")).map_or(key, |at| &key[..at])
}

/// The key of a table name at a version: the name, then the version's marker.
pub(crate) fn with_version(mut name: Vec<String>, version: Option<&VersionSpec>) -> Vec<String> {
    name.extend(version.map(VersionSpec::marker));
    name
}
