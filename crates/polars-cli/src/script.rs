use std::ops::Range;

use sqlparser::dialect::GenericDialect;
use sqlparser::tokenizer::{Location as TokenLocation, Token, TokenWithSpan, Tokenizer};

use crate::error::{ErrorRecord, Location};

/// A SQL script, split into its statements.
///
/// Statements are separated by `;`. Text that holds only whitespace and comments is not a
/// statement.
pub struct Script {
    text: String,
    statements: Vec<Range<usize>>,
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
                (format!("sql parser error: {error}"), location)
            });

        // A statement is the text between two separators.
        let mut cursor = Cursor::new(&text);
        let mut statements = Vec::new();
        let mut start = 0;
        let mut has_tokens = false;
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
                        statements.push(trimmed(&text, start..offset));
                    }
                    start = offset + 1;
                    has_tokens = false;
                },
                _ => has_tokens = true,
            }
        }
        // With an error, the text after the last separator is an incomplete statement.
        if error.is_none() && has_tokens {
            statements.push(trimmed(&text, start..text.len()));
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

    /// The statements that can be run, in order, each without its `;`.
    pub(crate) fn statements(&self) -> impl Iterator<Item = &str> {
        self.statements
            .iter()
            .map(|statement| &self.text[statement.clone()])
    }

    /// The number of statements, counting the one that cannot be split.
    pub(crate) fn len(&self) -> usize {
        self.statements.len() + usize::from(self.error.is_some())
    }

    /// The error of the statement that cannot be split. It follows all other statements.
    pub(crate) fn split_error(&self) -> Option<ErrorRecord> {
        let (message, location) = self.error.as_ref()?;
        Some(ErrorRecord {
            class: "SQLInterfaceError",
            message: message.clone(),
            hint: None,
            location: Some(*location),
            statement: self.len(),
        })
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
        script.statements().map(str::to_string).collect()
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
        assert_eq!(script.statements().collect::<Vec<_>>(), ["SELECT 1"]);
        assert!(!script.is_empty());
        assert_eq!(script.len(), 2);

        let error = script.split_error().unwrap();
        assert_eq!(error.class, "SQLInterfaceError");
        assert_eq!(error.statement, 2);
        assert_eq!(error.location, Some(Location { line: 2, column: 8 }));
        assert!(
            error
                .message
                .starts_with("sql parser error: Unterminated string literal"),
            "{}",
            error.message
        );
    }
}
