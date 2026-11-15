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
    /// A fraction of a second, with its strftime spec for writing.
    Fraction(&'static str),
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
    ("FF3", Item::Fraction("%3f")),
    ("FF4", Item::Unsupported),
    ("FF5", Item::Unsupported),
    ("FF6", Item::Fraction("%6f")),
    ("TZH", Item::Unsupported),
    ("TZM", Item::Unsupported),
    ("HH", Item::Number('I')),
    ("MI", Item::Number('M')),
    ("SS", Item::Number('S')),
    ("MS", Item::Fraction("%3f")),
    ("US", Item::Fraction("%6f")),
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

/// A template as strftime parts, with the fields that parsing needs to know about.
#[derive(Default)]
struct Template {
    parts: Vec<Part>,
    hour: bool,
    hour12: bool,
    minute: bool,
    second: bool,
    fraction: bool,
    meridian: bool,
    iso_week: bool,
    iso_weekday: bool,
}

/// The parts of `template`; `output` is true for `TO_CHAR` and false for parsing.
fn template_parts(template: &str, function: &str, output: bool) -> PolarsResult<Template> {
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

    let mut out = Template::default();
    let parts = &mut out.parts;
    let mut fractions = 0;
    let mut fill = false;
    let mut rest = template;
    while let Some(c) = rest.chars().next() {
        if c == '"' {
            // Quoted text is copied; a backslash takes the next character as is.
            let mut chars = rest[1..].char_indices();
            let mut end = rest.len();
            while let Some((i, c)) = chars.next() {
                match c {
                    '"' => {
                        end = i + 2;
                        break;
                    },
                    '\\' => {
                        if let Some((_, c)) = chars.next() {
                            push_literal(parts, c);
                        }
                    },
                    c => push_literal(parts, c),
                }
            }
            rest = &rest[end..];
            fill = false;
            continue;
        }
        if rest.starts_with("\\\"") {
            push_literal(parts, '"');
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
            push_literal(parts, c);
            rest = &rest[c.len_utf8()..];
            fill = false;
            continue;
        };
        match item {
            Item::Number(spec) => {
                match spec {
                    'H' => out.hour = true,
                    'I' => (out.hour, out.hour12) = (true, true),
                    'M' => out.minute = true,
                    'S' => out.second = true,
                    'V' => out.iso_week = true,
                    _ => {},
                }
                if fill {
                    push_str(parts, &format!("%-{spec}"))
                } else {
                    push_str(parts, &format!("%{spec}"))
                }
            },
            Item::Fixed(spec) => {
                out.iso_weekday |= spec == "%u";
                push_str(parts, spec)
            },
            Item::Fraction(spec) if output => push_str(parts, spec),
            Item::Fraction(spec) => {
                out.fraction = true;
                fractions += 1;
                polars_ensure!(
                    fractions == 1,
                    SQLInterface: "{} does not support more than one fraction of a second", function
                );
                // After a dot, any number of digits is a fraction, as in PostgreSQL.
                match parts.last_mut() {
                    Some(Part::Strftime(fmt)) if fmt.ends_with('.') => {
                        fmt.pop();
                        fmt.push_str("%.f");
                    },
                    _ => push_str(parts, spec),
                }
            },
            Item::Meridian => {
                out.meridian = true;
                if output && case == Case::Lower {
                    push_str(parts, "%P")
                } else {
                    push_str(parts, "%p")
                }
            },
            Item::Name(spec, padded) => {
                out.iso_weekday |= matches!(spec, "%A" | "%a");
                let pad = padded && !fill;
                if !output {
                    push_str(parts, spec);
                    // A space reads the padding, if any.
                    if pad {
                        push_str(parts, " ");
                    }
                } else if case == Case::Title && !pad {
                    push_str(parts, spec);
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
    Ok(out)
}

/// How to parse text written with a PostgreSQL template.
pub(crate) struct ParseFormat {
    pub(crate) format: String,
    /// Text to add to each value for the fields that PostgreSQL fills in when the template has
    /// none: AM for a 12-hour clock, Monday for an ISO week, and 0 for the hour, minute or
    /// second of a time.
    pub(crate) defaults: String,
}

/// The strftime format that parses text written with a PostgreSQL `template`. A template with
/// `%` in it is already a strftime format.
pub(crate) fn parse_format(template: &str, function: &str) -> PolarsResult<ParseFormat> {
    let mut out = ParseFormat {
        format: template.to_string(),
        defaults: String::new(),
    };
    if template.contains('%') {
        return Ok(out);
    }
    let template = template_parts(template, function, false)?;
    out.format = template
        .parts
        .into_iter()
        .map(|part| match part {
            Part::Strftime(fmt) => fmt,
            Part::Name { spec, .. } => spec.to_string(),
        })
        .collect();
    if template.hour12 && !template.meridian {
        out.format.push_str(" %p");
        out.defaults.push_str(" AM");
    }
    if template.iso_week && !template.iso_weekday {
        out.format.push_str(" %u");
        out.defaults.push_str(" 1");
    }
    // Parsing a time needs its hour and minute, and a fraction needs its second.
    let time = template.hour || template.minute || template.second || template.fraction;
    if template.fraction && !template.second {
        out.format.push_str(" %S");
        out.defaults.push_str(" 00");
    }
    if time && !template.hour {
        out.format.push_str(" %H");
        out.defaults.push_str(" 00");
    }
    if time && !template.minute {
        out.format.push_str(" %M");
        out.defaults.push_str(" 00");
    }
    Ok(out)
}

/// `expr` (a date, datetime or time) written as text with a PostgreSQL `template`. A template
/// with `%` in it is a strftime format.
pub(crate) fn format_with_template(expr: Expr, template: &str) -> PolarsResult<Expr> {
    if template.contains('%') {
        return Ok(expr.dt().strftime(template));
    }
    let mut pieces = template_parts(template, "TO_CHAR", true)?
        .parts
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
