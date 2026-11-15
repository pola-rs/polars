//! PostgreSQL template patterns, as used by `TO_CHAR`, `TO_DATE` and `TO_TIMESTAMP`.

use polars_core::prelude::*;
use polars_lazy::prelude::*;

/// What a template pattern stands for.
#[derive(Clone, Copy)]
enum Item {
    /// A number written with the strftime spec `%c`, zero padded unless after `FM`.
    Number(char),
    /// A strftime spec that `FM` does not change.
    Fixed(&'static str),
    /// A day or month name; `true` if it is padded to 9 characters unless after `FM`.
    Name(&'static str, bool),
    Meridian,
    Unsupported,
}

/// The template patterns, longer ones first so that the longest one matches.
const PATTERNS: &[(&str, Item)] = &[
    ("Y,YYY", Item::Unsupported),
    ("SSSSS", Item::Unsupported),
    ("MONTH", Item::Name("%B", true)),
    ("SSSS", Item::Unsupported),
    ("HH24", Item::Number('H')),
    ("HH12", Item::Number('I')),
    ("IDDD", Item::Unsupported),
    ("IYYY", Item::Number('G')),
    ("YYYY", Item::Number('Y')),
    ("A.M.", Item::Unsupported),
    ("P.M.", Item::Unsupported),
    ("A.D.", Item::Unsupported),
    ("B.C.", Item::Unsupported),
    ("IYY", Item::Unsupported),
    ("YYY", Item::Unsupported),
    ("DDD", Item::Number('j')),
    ("DAY", Item::Name("%A", true)),
    ("MON", Item::Name("%b", false)),
    ("FF1", Item::Unsupported),
    ("FF2", Item::Unsupported),
    ("FF3", Item::Fixed("%3f")),
    ("FF4", Item::Unsupported),
    ("FF5", Item::Unsupported),
    ("FF6", Item::Fixed("%6f")),
    ("TZH", Item::Unsupported),
    ("TZM", Item::Unsupported),
    ("HH", Item::Number('I')),
    ("MI", Item::Number('M')),
    ("SS", Item::Number('S')),
    ("MS", Item::Fixed("%3f")),
    ("US", Item::Fixed("%6f")),
    ("AM", Item::Meridian),
    ("PM", Item::Meridian),
    ("YY", Item::Number('y')),
    ("IY", Item::Number('g')),
    ("MM", Item::Number('m')),
    ("DD", Item::Number('d')),
    ("DY", Item::Name("%a", false)),
    ("ID", Item::Fixed("%u")),
    ("IW", Item::Number('V')),
    ("WW", Item::Unsupported),
    ("CC", Item::Unsupported),
    ("RM", Item::Unsupported),
    ("AD", Item::Unsupported),
    ("BC", Item::Unsupported),
    ("TZ", Item::Unsupported),
    ("OF", Item::Unsupported),
    ("TH", Item::Unsupported),
    ("FX", Item::Unsupported),
    ("TM", Item::Unsupported),
    ("Y", Item::Unsupported),
    ("I", Item::Unsupported),
    ("D", Item::Unsupported),
    ("W", Item::Unsupported),
    ("J", Item::Unsupported),
    ("Q", Item::Unsupported),
];

#[derive(Clone, Copy, PartialEq)]
enum Case {
    Upper,
    Title,
    Lower,
}

/// A piece of the text that `TO_CHAR` writes.
enum Part {
    Strftime(String),
    /// A name that strftime cannot write as asked: not in title case, or padded.
    Name {
        spec: &'static str,
        case: Case,
        pad: bool,
    },
}

/// The pattern at the start of `rest`, with its length and case. Patterns match in upper or
/// lower case, and names also in title case (`Month`), as in PostgreSQL.
fn match_pattern(rest: &str) -> Option<(usize, Item, Case)> {
    PATTERNS.iter().find_map(|&(pattern, item)| {
        let head = rest.get(..pattern.len())?;
        if !head.eq_ignore_ascii_case(pattern) {
            return None;
        }
        let is_lower = |s: &str| s.bytes().all(|b| !b.is_ascii_uppercase());
        let case = if head == pattern {
            Case::Upper
        } else if is_lower(head) {
            Case::Lower
        } else if matches!(item, Item::Name(..))
            && head.as_bytes()[0].is_ascii_uppercase()
            && is_lower(&head[1..])
        {
            Case::Title
        } else {
            return None;
        };
        Some((pattern.len(), item, case))
    })
}

/// The parts of `template`; `output` is true for `TO_CHAR` and false for parsing.
fn template_parts(template: &str, function: &str, output: bool) -> PolarsResult<Vec<Part>> {
    fn push_str(parts: &mut Vec<Part>, s: &str) {
        match parts.last_mut() {
            Some(Part::Strftime(fmt)) => fmt.push_str(s),
            _ => parts.push(Part::Strftime(s.to_string())),
        }
    }
    fn push_literal(parts: &mut Vec<Part>, c: char) {
        match c {
            '%' => push_str(parts, "%%"),
            c => push_str(parts, c.encode_utf8(&mut [0; 4])),
        }
    }

    let mut parts = vec![];
    let mut fill = false;
    let mut rest = template;
    while let Some(c) = rest.chars().next() {
        if c == '"' {
            // Quoted text is copied; `\"` is a quote.
            let mut chars = rest[1..].char_indices();
            let mut end = rest.len();
            while let Some((i, c)) = chars.next() {
                match c {
                    '"' => {
                        end = i + 2;
                        break;
                    },
                    '\\' if rest[1 + i + 1..].starts_with('"') => {
                        push_literal(&mut parts, '"');
                        chars.next();
                    },
                    c => push_literal(&mut parts, c),
                }
            }
            rest = &rest[end..];
            fill = false;
            continue;
        }
        if rest.starts_with("\\\"") {
            push_literal(&mut parts, '"');
            rest = &rest[2..];
            fill = false;
            continue;
        }
        if rest.starts_with("FM") || rest.starts_with("fm") {
            fill = true;
            rest = &rest[2..];
            continue;
        }
        let Some((len, item, case)) = match_pattern(rest) else {
            push_literal(&mut parts, c);
            rest = &rest[c.len_utf8()..];
            fill = false;
            continue;
        };
        match item {
            Item::Number(spec) if fill => push_str(&mut parts, &format!("%-{spec}")),
            Item::Number(spec) => push_str(&mut parts, &format!("%{spec}")),
            Item::Fixed(spec) => push_str(&mut parts, spec),
            Item::Meridian if output && case == Case::Lower => push_str(&mut parts, "%P"),
            Item::Meridian => push_str(&mut parts, "%p"),
            Item::Name(spec, padded) => {
                let pad = padded && !fill;
                if !output || (case == Case::Title && !pad) {
                    push_str(&mut parts, spec);
                } else {
                    parts.push(Part::Name { spec, case, pad });
                }
            },
            Item::Unsupported => {
                polars_bail!(SQLInterface: "{} does not support the template pattern '{}'", function, &rest[..len])
            },
        }
        rest = &rest[len..];
        fill = false;
    }
    Ok(parts)
}

/// The strftime format that parses text written with a PostgreSQL `template`. A template with
/// `%` in it is already a strftime format.
pub(crate) fn parse_format(template: &str, function: &str) -> PolarsResult<String> {
    if template.contains('%') {
        return Ok(template.to_string());
    }
    Ok(template_parts(template, function, false)?
        .into_iter()
        .map(|part| match part {
            Part::Strftime(fmt) => fmt,
            Part::Name { spec, .. } => spec.to_string(),
        })
        .collect())
}

/// `expr` (a date, datetime or time) written as text with a PostgreSQL `template`. A template
/// with `%` in it is a strftime format.
pub(crate) fn format_with_template(expr: Expr, template: &str) -> PolarsResult<Expr> {
    if template.contains('%') {
        return Ok(expr.dt().strftime(template));
    }
    let mut pieces = template_parts(template, "TO_CHAR", true)?
        .into_iter()
        .map(|part| match part {
            Part::Strftime(fmt) => expr.clone().dt().strftime(&fmt),
            Part::Name { spec, case, pad } => {
                let name = expr.clone().dt().strftime(spec);
                let name = match case {
                    Case::Upper => name.str().to_uppercase(),
                    Case::Lower => name.str().to_lowercase(),
                    Case::Title => name,
                };
                if pad {
                    name.str().pad_end(lit(9), ' ')
                } else {
                    name
                }
            },
        })
        .collect::<Vec<_>>();
    Ok(match pieces.len() {
        0 => expr.dt().strftime(""),
        1 => pieces.pop().unwrap(),
        _ => concat_str(pieces, "", false),
    })
}
