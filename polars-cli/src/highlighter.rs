use std::io::{self};

use nu_ansi_term::{Color, Style};
use polars::sql::keywords::{all_functions, all_keywords};
use reedline::{Highlighter, StyledText};
use sqlparser::dialect::GenericDialect;
use sqlparser::keywords::Keyword;
use sqlparser::tokenizer::{Token, Tokenizer};

use crate::interactive::PolarsCommand;

pub(crate) struct SQLHighlighter {}

fn colorize_sql(query: &str, st: &mut StyledText) -> std::io::Result<()> {
    let dialect = GenericDialect;

    let mut tokenizer = Tokenizer::new(&dialect, query);

    let tokens = tokenizer.tokenize().map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Failed to tokenize SQL: {}", e),
        )
    });

    // the tokenizer will error if the final character is an unescaped quote
    // such as `select * from read_csv("
    // in this case we will try to find the quote and colorize the rest of the query
    // otherwise we will return the error
    if let Err(err) = tokens {
        let pos = query.find(&['\'', '"'][..]);
        match pos {
            None => return Err(err),
            Some(pos) => {
                let (s1, s2) = query.split_at(pos);
                colorize_sql(s1, st)?;

                st.push((Style::new(), s2.to_string()));
                return Ok(());
            }
        }
    }
    let tokens = tokens.unwrap();

    for token in tokens {
        match token {
            Token::Mul => st.push((Style::new().fg(Color::Purple), "*".to_string())),
            Token::LParen => st.push((Style::new().fg(Color::Purple), "(".to_string())),
            Token::RParen => st.push((Style::new().fg(Color::Purple), ")".to_string())),
            Token::Comma => st.push((Style::new().fg(Color::Purple), ",".to_string())),
            Token::SemiColon => st.push((Style::new().fg(Color::White).bold(), ";".to_string())),
            Token::SingleQuotedString(s) => {
                st.push((Style::new().fg(Color::Yellow).italic(), format!("'{}'", s)))
            }
            Token::DoubleQuotedString(s) => st.push((
                Style::new().fg(Color::Yellow).italic(),
                format!("\"{}\"", s),
            )),
            Token::Word(w) => match w.keyword {
                Keyword::SELECT
                | Keyword::FROM
                | Keyword::WHERE
                | Keyword::GROUP
                | Keyword::BY
                | Keyword::ORDER
                | Keyword::LIMIT
                | Keyword::OFFSET
                | Keyword::AND
                | Keyword::OR
                | Keyword::AS
                | Keyword::ON
                | Keyword::INNER
                | Keyword::LEFT
                | Keyword::RIGHT
                | Keyword::FULL
                | Keyword::OUTER
                | Keyword::JOIN
                | Keyword::CREATE
                | Keyword::TABLE
                | Keyword::SHOW
                | Keyword::TABLES
                | Keyword::VARCHAR
                | Keyword::INT
                | Keyword::FLOAT
                | Keyword::DOUBLE
                | Keyword::BOOLEAN
                | Keyword::DATE
                | Keyword::TIME
                | Keyword::DATETIME
                | Keyword::ARRAY
                | Keyword::ASC
                | Keyword::DESC
                | Keyword::NULL
                | Keyword::NOT
                | Keyword::IN
                | Keyword::WITH => {
                    st.push((Style::new().fg(Color::LightGreen), format!("{w}")));
                }
                _ => match w.to_string().as_str() {
                    s if all_functions().contains(&s) => {
                        st.push((Style::new().fg(Color::LightCyan).bold(), format!("{w}")))
                    }
                    s if all_keywords().contains(&s) => {
                        st.push((Style::new().fg(Color::LightGreen), format!("{w}")))
                    }
                    s if PolarsCommand::keywords().contains(&s) => {
                        st.push((Style::new().fg(Color::LightGray).bold(), format!("{w}")))
                    }
                    _ => st.push((Style::new(), format!("{w}"))),
                },
            },
            other => {
                st.push((Style::new(), format!("{other}")));
            }
        }
    }
    Ok(())
}

impl Highlighter for SQLHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> reedline::StyledText {
        let mut styled_text = StyledText::new();
        colorize_sql(line, &mut styled_text).unwrap();
        styled_text
    }
}
