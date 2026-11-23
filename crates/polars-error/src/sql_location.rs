use std::fmt::Write;

/// A position in SQL text, counted from 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqlPosition {
    pub line: u64,
    pub column: u64,
}

/// How precisely a location points at the cause.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SqlLocationKind {
    /// The span is the cause.
    Exact,
    /// The span is the clause the cause is in.
    Clause(String),
    /// Only the statement is known.
    Statement,
}

/// Where in the SQL text an error is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SqlLocation {
    pub kind: SqlLocationKind,
    pub start: SqlPosition,
    pub end: SqlPosition,
    /// The text of the line of `start`; empty until known.
    pub line: String,
}

/// Lines longer than this are cut around the position.
const MAX_WIDTH: usize = 100;

impl SqlLocation {
    /// The statement `sql` on one line, without its comments, at its first token.
    pub fn statement(sql: &str) -> Self {
        let mut text = String::new();
        let mut start = None;
        let mut position = SqlPosition { line: 1, column: 1 };
        let advance = |position: &mut SqlPosition, c: char| match c {
            '\n' => {
                *position = SqlPosition {
                    line: position.line + 1,
                    column: 1,
                }
            },
            _ => position.column += 1,
        };
        let mut chars = sql.chars().peekable();
        let mut quote = None;
        let mut space = false;
        while let Some(c) = chars.next() {
            let at = position;
            advance(&mut position, c);
            if quote.is_some() {
                text.push(c);
                if quote == Some(c) {
                    quote = None;
                }
                continue;
            }
            match (c, chars.peek()) {
                ('-', Some('-')) => {
                    while let Some(c) = chars.next_if(|c| *c != '\n') {
                        advance(&mut position, c);
                    }
                    space = true;
                },
                ('/', Some('*')) => {
                    let mut previous = ' ';
                    for c in chars.by_ref() {
                        advance(&mut position, c);
                        if previous == '*' && c == '/' {
                            break;
                        }
                        previous = c;
                    }
                    space = true;
                },
                (c, _) if c.is_whitespace() => space = true,
                (c, _) => {
                    match start {
                        None => start = Some(at),
                        Some(_) if space => text.push(' '),
                        Some(_) => {},
                    }
                    space = false;
                    if c == '\'' || c == '"' {
                        quote = Some(c);
                    }
                    text.push(c);
                },
            }
        }
        let start = start.unwrap_or(SqlPosition { line: 1, column: 1 });
        Self {
            kind: SqlLocationKind::Statement,
            start,
            end: start,
            line: text,
        }
    }

    pub(crate) fn write_after_message(&self, out: &mut String) {
        let SqlPosition { line, column } = self.start;
        match &self.kind {
            SqlLocationKind::Exact if !self.line.is_empty() => {
                let (excerpt, caret) = excerpt(&self.line, column.saturating_sub(1) as usize);
                let label = format!("LINE {line}: ");
                let indent = " ".repeat(label.len() + caret);
                write!(
                    out,
                    "\n\nSQL at line {line}, column {column}:\n{label}{excerpt}\n{indent}^"
                )
            },
            SqlLocationKind::Exact => write!(out, "\n\nSQL at line {line}, column {column}."),
            SqlLocationKind::Clause(clause) => write!(
                out,
                "\n\nIn the {clause} clause at line {line}, column {column}."
            ),
            SqlLocationKind::Statement if self.line.is_empty() => Ok(()),
            SqlLocationKind::Statement => {
                write!(
                    out,
                    "\n\nIn the SQL statement: {}",
                    excerpt(&self.line, 0).0
                )
            },
        }
        .unwrap()
    }
}

/// The excerpt of `line` around `at`, and where `at` is in it.
fn excerpt(line: &str, at: usize) -> (String, usize) {
    // One space per tab or other control character keeps the caret aligned.
    let chars: Vec<char> = line
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let at = at.min(chars.len());
    if chars.len() <= MAX_WIDTH {
        return (chars.into_iter().collect(), at);
    }
    let start = at
        .saturating_sub(MAX_WIDTH / 2)
        .min(chars.len() - MAX_WIDTH);
    let end = start + MAX_WIDTH;
    let mut out = String::new();
    let mut at = at - start;
    if start > 0 {
        out.push_str("...");
        at += 3;
    }
    out.extend(&chars[start..end]);
    if end < chars.len() {
        out.push_str("...");
    }
    (out, at)
}
