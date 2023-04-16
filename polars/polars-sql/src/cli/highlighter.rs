use std::io::Cursor;

use nu_ansi_term::{Color, Style as AnsiStyle};
use reedline::{Highlighter, StyledText};
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style, Theme, ThemeSet};
use syntect::parsing::{SyntaxDefinition, SyntaxSet, SyntaxSetBuilder};
use syntect::util::as_24_bit_terminal_escaped;

pub(crate) struct SQLHighlighter {}

const SQL_SYNTAX: &str = include_str!("../../assets/SQL.sublime-syntax");
const THEME: &[u8] = include_bytes!("../../assets/theme");

impl Highlighter for SQLHighlighter {
    fn highlight(&self, line: &str, cursor: usize) -> reedline::StyledText {
        let syn_def = SyntaxDefinition::load_from_str(SQL_SYNTAX, true, Some("sql")).unwrap();
        let mut ssb = SyntaxSetBuilder::new();
        ssb.add(syn_def);
        let mut ps = ssb.build();
        let mut cursor = Cursor::new(THEME);
        let ts = ThemeSet::load_from_reader(&mut cursor).unwrap();
        let syntax = ps.find_syntax_by_extension("sql").unwrap();

        let mut styled_text = StyledText::new();
        let mut h = HighlightLines::new(syntax, &ts);
        let ranges: Vec<(Style, &str)> = h.highlight_line(line, &ps).unwrap();
        for (style, text) in ranges {
            let fg = Color::Rgb(style.foreground.r, style.foreground.g, style.foreground.b);
            let s = AnsiStyle::new().fg(fg);
            styled_text.push((s, text.to_string()));
        }
        styled_text
    }
}
