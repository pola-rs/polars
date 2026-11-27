use std::ops::Range;

use sqlparser::dialect::GenericDialect;
use sqlparser::keywords::Keyword;
use sqlparser::tokenizer::{Location as TokenLocation, Token, TokenWithSpan, Tokenizer, Word};

use crate::error::{ErrorRecord, Location};

/// A statement of a script, without its `;`.
pub(crate) struct Statement<'a> {
    pub sql: &'a str,
    /// Where it starts in the script, in bytes.
    pub offset: usize,
    /// What the statement returns.
    pub kind: StatementKind,
}

/// What the result of a statement is. The engine does not tell what kind of statement a
/// result comes from, so the first keyword of the statement stands in for that, and `TO
/// STDOUT` in a `COPY`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StatementKind {
    /// Rows, as a query and `COPY ... TO STDOUT` return.
    #[default]
    Query,
    /// A description of a table, a query or a plan, as `DESCRIBE` and `EXPLAIN` return. It is
    /// read as a whole.
    Description,
    /// None: the statement changes the session or writes a file, as `CREATE` and `COPY ... TO`
    /// a file do.
    Effect,
}

/// A SQL script, split into its statements.
///
/// Statements are separated by `;`. Text that holds only whitespace and comments is not a
/// statement.
pub struct Script {
    text: String,
    statements: Vec<(Range<usize>, StatementKind)>,
    /// Why the text after the last statement cannot be split, for example an unclosed string.
    error: Option<(String, Location)>,
}

impl Script {
    pub fn new(text: String) -> Self {
        // The dialect of `polars-sql`, so that both read strings and comments alike.
        let mut tokens = Vec::new();
        let error = Tokenizer::new(&GenericDialect, &text)
            .tokenize_with_location_into_buf(&mut tokens)
            .err()
            .map(|error| {
                let location = Location {
                    line: error.location.line as usize,
                    column: error.location.column as usize,
                };
                (format!("sql parser error: {}", error.message), location)
            });

        // A statement is the text between two separators.
        let mut cursor = Cursor::new(&text);
        let mut statements = Vec::new();
        let mut start = 0;
        let mut has_tokens = false;
        let mut kind = KindReader::default();
        // Where the executable comment that is being passed ends.
        let mut comment_end = 0;
        for TokenWithSpan { token, span } in &tokens {
            let offset = cursor.offset_of(span.start);
            // The tokenizer reports the content of an executable comment (`/*! ... */`) as
            // tokens of the statement, the first of them at the start of the comment. A `;`
            // among them does not separate statements; the parser decides what it means.
            if offset >= comment_end && text[offset..].starts_with("/*!") {
                comment_end = offset + block_comment_len(&text[offset..]);
            }
            let in_comment = offset < comment_end;
            match token {
                Token::Whitespace(_) | Token::EOF => {},
                Token::SemiColon if !in_comment => {
                    if has_tokens {
                        statements.push((trimmed(&text, start..offset), kind.kind()));
                    }
                    start = offset + 1;
                    has_tokens = false;
                    kind = KindReader::default();
                },
                _ => {
                    kind.read(token);
                    has_tokens = true;
                },
            }
        }
        // With an error, the text after the last separator is an incomplete statement.
        if error.is_none() && has_tokens {
            statements.push((trimmed(&text, start..text.len()), kind.kind()));
        }

        Self {
            text,
            statements,
            error,
        }
    }

    /// Whether the script has nothing to run and nothing to report.
    pub fn is_empty(&self) -> bool {
        self.statements.is_empty() && self.error.is_none()
    }

    /// The statements that can be run, in order.
    pub(crate) fn statements(&self) -> impl Iterator<Item = Statement<'_>> {
        self.statements.iter().map(|(range, kind)| Statement {
            sql: &self.text[range.clone()],
            offset: range.start,
            kind: *kind,
        })
    }

    /// Line `number` of the script, counted from 1.
    pub(crate) fn line(&self, number: usize) -> Option<&str> {
        self.text.lines().nth(number.checked_sub(1)?)
    }

    /// The line and column of the byte `offset` of the script.
    pub(crate) fn location_of(&self, offset: usize) -> Location {
        let before = &self.text[..offset];
        let line_start = before.rfind('\n').map_or(0, |i| i + 1);
        Location {
            line: before.matches('\n').count() + 1,
            column: before[line_start..].chars().count() + 1,
        }
    }

    /// Whether the last statement is the only one that returns rows, as a format that writes
    /// one file needs. Text at the end that cannot be split is not a statement.
    pub fn ends_in_its_only_result(&self) -> bool {
        let mut kinds = self.statements.iter().rev().map(|(_, kind)| *kind);
        kinds
            .next()
            .is_some_and(|last| last != StatementKind::Effect)
            && kinds.all(|kind| kind == StatementKind::Effect)
    }

    /// Whether all of the text splits into statements.
    pub fn can_be_split(&self) -> bool {
        self.error.is_none()
    }

    /// The number of statements, counting the one that cannot be split.
    pub fn len(&self) -> usize {
        self.statements.len() + usize::from(self.error.is_some())
    }

    /// The error of the statement that cannot be split. It follows all other statements.
    pub(crate) fn split_error(&self) -> Option<ErrorRecord> {
        let (message, location) = self.error.as_ref()?;
        Some(ErrorRecord::split(
            message.clone(),
            *location,
            self.len(),
            self,
        ))
    }
}

/// Reads the kind of a statement from its tokens, in order.
#[derive(Default)]
struct KindReader {
    /// The kind that the first word gives. An executable comment can start with a `;`, so it is
    /// the first word, not the first token.
    first: Option<StatementKind>,
    copy: bool,
    /// The depth of parentheses, and whether the last token was `TO` outside them.
    depth: usize,
    after_to: bool,
    to_stdout: bool,
}

impl KindReader {
    fn read(&mut self, token: &Token) {
        if self.first.is_none()
            && let Token::Word(word) = token
        {
            self.first = Some(kind_of(word));
            self.copy = word.keyword == Keyword::COPY;
        }
        let after_to = std::mem::take(&mut self.after_to);
        match token {
            Token::LParen => self.depth += 1,
            Token::RParen => self.depth = self.depth.saturating_sub(1),
            Token::Word(word) if self.depth == 0 => {
                self.to_stdout |= after_to && word.keyword == Keyword::STDOUT;
                self.after_to = word.keyword == Keyword::TO;
            },
            _ => {},
        }
    }

    fn kind(&self) -> StatementKind {
        if self.copy && self.to_stdout {
            StatementKind::Query
        } else {
            self.first.unwrap_or_default()
        }
    }
}

/// The kind of a statement whose first word is `word`.
fn kind_of(word: &Word) -> StatementKind {
    match word.keyword {
        Keyword::DESCRIBE | Keyword::DESC | Keyword::EXPLAIN => StatementKind::Description,
        Keyword::ALTER
        | Keyword::ATTACH
        | Keyword::COPY
        | Keyword::CREATE
        | Keyword::DELETE
        | Keyword::DETACH
        | Keyword::DROP
        | Keyword::INSERT
        | Keyword::MERGE
        | Keyword::REPLACE
        | Keyword::TRUNCATE
        | Keyword::UPDATE
        | Keyword::USE => StatementKind::Effect,
        _ => StatementKind::Query,
    }
}

/// The length of the block comment that `text` starts with. Block comments nest.
fn block_comment_len(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut depth = 0;
    let mut i = 0;
    while i + 1 < bytes.len() {
        match &bytes[i..i + 2] {
            b"/*" => {
                depth += 1;
                i += 2;
            },
            b"*/" => {
                depth -= 1;
                i += 2;
                if depth == 0 {
                    return i;
                }
            },
            _ => i += 1,
        }
    }
    text.len()
}

/// `range` without the whitespace at its ends.
fn trimmed(text: &str, range: Range<usize>) -> Range<usize> {
    let statement = &text[range.clone()];
    let start = range.start + (statement.len() - statement.trim_start().len());
    start..start + statement.trim().len()
}

/// Maps the positions of tokens, which are lines and columns, to byte offsets in the text.
///
/// It only moves forward, so that a script is read once however many statements it holds.
struct Cursor<'a> {
    text: &'a str,
    offset: usize,
    line: u64,
    column: u64,
}

impl<'a> Cursor<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            offset: 0,
            line: 1,
            column: 1,
        }
    }

    /// The offset of `location`, which must not lie before the previous one asked for.
    fn offset_of(&mut self, location: TokenLocation) -> usize {
        let mut rest = self.text[self.offset..].chars();
        while (self.line, self.column) < (location.line, location.column) {
            // Lines and columns are counted as the tokenizer counts them.
            match rest.next() {
                Some('\n') => {
                    self.line += 1;
                    self.column = 1;
                    self.offset += 1;
                },
                Some(c) => {
                    self.column += 1;
                    self.offset += c.len_utf8();
                },
                None => break,
            }
        }
        self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str) -> Vec<String> {
        let script = Script::new(text.to_string());
        assert!(script.split_error().is_none());
        split_sql(&script)
    }

    fn split_sql(script: &Script) -> Vec<String> {
        script
            .statements()
            .map(|statement| statement.sql.to_string())
            .collect()
    }

    #[test]
    fn statement_kinds_follow_the_first_keyword() {
        use StatementKind::*;
        for (sql, kind) in [
            ("desc t", Description),
            ("-- how it runs\nEXPLAIN SELECT 1", Description),
            ("SELECT 'DESCRIBE'", Query),
            ("\"DESC\"", Query),
            ("(SELECT 1)", Query),
            ("create table t AS SELECT 1", Effect),
            ("COPY t TO 'f.csv'", Effect),
            ("/*! ; */ CREATE TABLE t AS SELECT 1", Effect),
            ("COPY (SELECT 1 AS \"to\") TO STDOUT (FORMAT csv)", Query),
            ("COPY t TO 'stdout'", Effect),
            ("REPLACE INTO t VALUES (2)", Effect),
        ] {
            let script = Script::new(sql.to_string());
            assert_eq!(script.statements().next().unwrap().kind, kind, "{sql}");
        }
    }

    #[test]
    fn a_one_file_format_takes_one_result_last() {
        for (sql, takes) in [
            ("CREATE TABLE t AS SELECT 1; SELECT * FROM t", true),
            (
                "CREATE TABLE t AS SELECT 1; COPY t TO STDOUT (FORMAT parquet)",
                true,
            ),
            ("CREATE TABLE t AS SELECT 1", false),
            ("SELECT 1; SELECT 2", false),
            ("SELECT 1; DROP TABLE t", false),
        ] {
            let script = Script::new(sql.to_string());
            assert_eq!(script.ends_in_its_only_result(), takes, "{sql}");
        }
    }

    #[test]
    fn statements_are_split_on_semicolons() {
        assert_eq!(split("SELECT 1"), ["SELECT 1"]);
        assert_eq!(split("SELECT 1;"), ["SELECT 1"]);
        assert_eq!(split("SELECT 1; SELECT 2"), ["SELECT 1", "SELECT 2"]);
        assert_eq!(
            split("  SELECT 1 ;\n\nSELECT\n  2 ;\n"),
            ["SELECT 1", "SELECT\n  2"]
        );
    }

    #[test]
    fn semicolons_in_strings_identifiers_and_comments_do_not_split() {
        assert_eq!(split("SELECT 'a;b' AS s"), ["SELECT 'a;b' AS s"]);
        assert_eq!(split("SELECT 'it''s; ok'"), ["SELECT 'it''s; ok'"]);
        assert_eq!(split(r#"SELECT 1 AS "a;b""#), [r#"SELECT 1 AS "a;b""#]);
        assert_eq!(
            split("SELECT 1 -- one; not two\n; SELECT 2"),
            ["SELECT 1 -- one; not two", "SELECT 2"]
        );
        assert_eq!(
            split("SELECT /* ; */ 1; SELECT 2"),
            ["SELECT /* ; */ 1", "SELECT 2"]
        );
    }

    #[test]
    fn text_without_a_statement_is_skipped() {
        assert_eq!(split(""), Vec::<String>::new());
        assert_eq!(
            split(" ;; \n -- only a comment\n /* and another */ ;"),
            Vec::<String>::new()
        );
        assert_eq!(split(";SELECT 1;;SELECT 2;;"), ["SELECT 1", "SELECT 2"]);
        assert!(Script::new("-- nothing".to_string()).is_empty());
        assert!(!Script::new("SELECT 1".to_string()).is_empty());
    }

    #[test]
    fn offsets_count_characters() {
        assert_eq!(
            split("SELECT 'é;ü' AS \"naïve\"; SELECT '日本'\n;SELECT 3"),
            ["SELECT 'é;ü' AS \"naïve\"", "SELECT '日本'", "SELECT 3"]
        );
        assert_eq!(
            split("SELECT 1;\r\nSELECT 2;\r\n"),
            ["SELECT 1", "SELECT 2"]
        );
    }

    #[test]
    fn executable_comments_stay_whole() {
        // The tokenizer reports the content of such a comment as tokens of the statement.
        assert_eq!(
            split("SELECT /*! 1 */ AS a; SELECT 2"),
            ["SELECT /*! 1 */ AS a", "SELECT 2"]
        );
        assert_eq!(
            split("SELECT 1 /*! AS a; */; SELECT 2"),
            ["SELECT 1 /*! AS a; */", "SELECT 2"]
        );
        assert_eq!(split("/*! SELECT 1 */"), ["/*! SELECT 1 */"]);
        // After a line break the tokens of the comment are where the text has them.
        assert_eq!(
            split("SELECT 1 /*!\n; */; SELECT 2"),
            ["SELECT 1 /*!\n; */", "SELECT 2"]
        );
        assert_eq!(
            split("SELECT 1 /*! /* ; */ ; */ AS a; SELECT ';' /*!50110 ; */;"),
            ["SELECT 1 /*! /* ; */ ; */ AS a", "SELECT ';' /*!50110 ; */"]
        );
        // Without content there is no statement.
        assert_eq!(split("/*! */; /*!50110*/ ;"), Vec::<String>::new());
    }

    #[test]
    fn many_statements_on_one_line() {
        let text = "SELECT 1;".repeat(20_000);
        assert_eq!(split(&text).len(), 20_000);
    }

    #[test]
    fn unclosed_string_fails_after_the_complete_statements() {
        let script = Script::new("SELECT 1;\nSELECT 'abc".to_string());
        assert_eq!(split_sql(&script), ["SELECT 1"]);
        assert!(!script.is_empty());
        assert_eq!(script.len(), 2);

        let error = script.split_error().unwrap();
        assert_eq!(error.class, "SQLInterfaceError");
        assert_eq!(error.statement, 2);
        let location = error.location.unwrap();
        assert_eq!(location.start, Location { line: 2, column: 8 });
        assert_eq!(location.line, "SELECT 'abc");
        assert!(
            error
                .message
                .starts_with("sql parser error: Unterminated string literal"),
            "{}",
            error.message
        );
    }
}
