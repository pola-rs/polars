//! Path strings for `pl.path`, e.g. `pl.path("a.b.c")`.
//!
//! A path is desugared into `struct.field` expressions, so it gets no dedicated IR node or
//! kernel.
//!
//! ```text
//! path     := name ( "." name )*
//! name     := ident | "`" escaped "`"
//! ident    := ( alpha | "_" ) ( alpha | numeric | "_" )*
//! escaped  := any chars, with `` for a literal backtick
//! ```
//!
//! Unquoted names are kept to letters, digits and `_`, and the only step is `.name`, so that other
//! characters (`[`, `*`, a leading digit as in `a.0`, ...) stay free for syntax added later.
use super::*;
use crate::constants::get_pl_element_name;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PathParseError {
    /// Character index into the path at which the error occurred.
    pub pos: usize,
    pub msg: String,
}

fn is_ident_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_ident_continue(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn err<T>(&self, pos: usize, msg: impl Into<String>) -> Result<T, PathParseError> {
        Err(PathParseError {
            pos,
            msg: msg.into(),
        })
    }

    fn name(&mut self) -> Result<PlSmallStr, PathParseError> {
        match self.peek() {
            None => self.err(self.pos, "expected a field name, found end of path"),
            Some('`') => {
                let start = self.pos;
                self.pos += 1;
                let mut name = String::new();
                loop {
                    match self.peek() {
                        None => return self.err(start, "unterminated backtick-quoted name"),
                        Some('`') if self.chars.get(self.pos + 1) == Some(&'`') => {
                            name.push('`');
                            self.pos += 2;
                        },
                        Some('`') => {
                            self.pos += 1;
                            return Ok(name.into());
                        },
                        Some(c) => {
                            name.push(c);
                            self.pos += 1;
                        },
                    }
                }
            },
            Some('.') => self.err(self.pos, "expected a field name, found `.`"),
            Some('*') => self.err(
                self.pos,
                "selecting all fields with `.*` is not supported yet",
            ),
            Some(c) if !is_ident_start(c) => self.err(
                self.pos,
                format!(
                    "an unquoted name must start with a letter or `_` (quote other names with backticks), found `{c}`"
                ),
            ),
            Some(_) => {
                let start = self.pos;
                while self.peek().is_some_and(is_ident_continue) {
                    self.pos += 1;
                }
                Ok(self.chars[start..self.pos]
                    .iter()
                    .collect::<String>()
                    .into())
            },
        }
    }
}

/// Parse a path into its names, the first being the root.
pub(super) fn parse_field_path(path: &str) -> Result<Vec<PlSmallStr>, PathParseError> {
    let mut p = Parser {
        chars: path.chars().collect(),
        pos: 0,
    };
    let mut names = vec![p.name()?];
    while let Some(c) = p.peek() {
        match c {
            '.' => {
                p.pos += 1;
                names.push(p.name()?);
            },
            '[' => {
                return p.err(
                    p.pos,
                    "`[` is reserved: a path only selects struct fields; use `.list` or `.arr` methods for list and array elements",
                );
            },
            '`' => return p.err(p.pos, "expected `.` or end of path, found `` ` ``"),
            c => {
                return p.err(
                    p.pos,
                    format!(
                        "an unquoted name may only contain letters, digits and `_` (quote other names with backticks), found `{c}`"
                    ),
                );
            },
        }
    }
    Ok(names)
}

fn write_name(out: &mut String, name: &str) {
    let mut chars = name.chars();
    if chars.next().is_some_and(is_ident_start) && chars.all(is_ident_continue) {
        out.push_str(name);
    } else {
        out.push('`');
        out.push_str(&name.replace('`', "``"));
        out.push('`');
    }
}

/// The output name and root of `pl.path(path)` without a schema: the last and first names of the
/// path, or `path` itself if it doesn't parse (the plan then fails with the parse error).
pub(crate) fn field_path_names(path: &PlSmallStr) -> (PlSmallStr, PlSmallStr) {
    match parse_field_path(path) {
        Ok(names) => (names.last().unwrap().clone(), names[0].clone()),
        Err(_) => (path.clone(), path.clone()),
    }
}

/// Resolve `pl.path(path)` into `struct.field` expressions.
///
/// Inside `struct.with_fields` the root is a field of the struct in scope, elsewhere it is a
/// column of the input schema. A column literally named `a.b` is `` pl.path("`a.b`") ``.
pub(super) fn field_path_to_expr(path: &PlSmallStr, ctx: &ExprToIRContext) -> PolarsResult<Expr> {
    // Inside `list.eval` / `arr.eval` the schema still holds the outer columns, but those can't
    // be used there. The schema check catches dtype resolution of a nested `eval`'s input, which
    // runs without the `in_eval` flag.
    polars_ensure!(
        ctx.with_fields.is_some() || !(ctx.in_eval || ctx.schema.contains(&get_pl_element_name())),
        InvalidOperation:
        "`pl.path({path:?})` is not allowed in `eval` outside of `struct.with_fields`; use `pl.element()` to refer to the element, e.g. `pl.element().struct.field(...)`"
    );

    // Point out a column or field whose name is the whole string.
    let literal_name_hint = || {
        let (what, func) = match &ctx.with_fields {
            Some(with_fields) if with_fields.contains(path) => ("field", "pl.field"),
            None if ctx.schema.contains(path) => ("column", "pl.col"),
            _ => return String::new(),
        };
        let mut quoted = String::new();
        write_name(&mut quoted, path);
        format!(
            "\n\nHint: to select the {what} with that exact name, use `{func}({path:?})` or quote it: `pl.path({quoted:?})`"
        )
    };

    let names = parse_field_path(path).map_err(|e| {
        polars_err!(
            InvalidOperation:
            "invalid path {path:?}: {} (at index {}){}", e.msg, e.pos, literal_name_hint()
        )
    })?;
    let (root, fields) = names.split_first().unwrap();

    let (mut expr, mut dtype) = match &ctx.with_fields {
        Some(with_fields) => {
            let Some(dtype) = with_fields.get(root) else {
                polars_bail!(
                    InvalidOperation:
                    "field `{root}` of path {path:?} does not exist on struct with fields {:?}",
                    with_fields.iter_names_cloned().collect::<Vec<_>>().as_slice()
                );
            };
            (Expr::Field([root.clone()].into()), dtype.clone())
        },
        None => {
            let dtype = match ctx.schema.get(root) {
                Some(dtype) => dtype.clone(),
                None if ctx.check_column_names => {
                    let hint = literal_name_hint();
                    return Err(ctx
                        .schema
                        .column_not_found_err(root)
                        .wrap_msg(|m| format!("{m}\n\n(resolving path {path:?}){hint}")));
                },
                None => DataType::Unknown(Default::default()),
            };
            (Expr::Column(root.clone()), dtype)
        },
    };

    // Rendered path up to the current name, for errors.
    let mut at = String::new();
    write_name(&mut at, root);

    for name in fields {
        dtype = match dtype {
            DataType::Struct(struct_fields) => {
                let Some(field) = struct_fields.iter().find(|f| f.name() == name) else {
                    polars_bail!(
                        StructFieldNotFound:
                        "field {name:?} not found in struct at {at:?} of path {path:?}; available fields: {:?}",
                        struct_fields.iter().map(|f| f.name().as_str()).collect::<Vec<_>>()
                    );
                };
                field.dtype().clone()
            },
            dtype @ DataType::Unknown(_) => dtype,
            dtype => polars_bail!(
                InvalidOperation:
                "cannot select field {name:?} of dtype {dtype} at {at:?} of path {path:?}; expected a struct"
            ),
        };
        // Built directly rather than with `struct.field`, so names like `*` aren't read as regex.
        expr = expr.map_unary(FunctionExpr::StructExpr(StructFunction::FieldByName(
            name.clone(),
        )));
        at.push('.');
        write_name(&mut at, name);
    }

    Ok(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Vec<String> {
        let names = parse_field_path(s).unwrap();
        names.iter().map(|n| n.to_string()).collect()
    }

    #[test]
    fn test_parse_field_path() {
        assert_eq!(parse("a"), ["a"]);
        assert_eq!(parse("a.b.c"), ["a", "b", "c"]);
        assert_eq!(parse("ä.ö"), ["ä", "ö"]);
        assert_eq!(parse("_.a_1._2"), ["_", "a_1", "_2"]);
    }

    #[test]
    fn test_parse_field_path_escaped() {
        assert_eq!(parse("`x.y`.`[0]`.``"), ["x.y", "[0]", ""]);
        assert_eq!(parse("`a``b`"), ["a`b"]);
        assert_eq!(parse("````"), ["`"]);
        assert_eq!(parse("a.`0`.b"), ["a", "0", "b"]);
    }

    #[test]
    fn test_parse_field_path_errors() {
        let start = |c: &str| {
            format!(
                "an unquoted name must start with a letter or `_` (quote other names with backticks), found `{c}`"
            )
        };
        let inner = |c: &str| {
            format!(
                "an unquoted name may only contain letters, digits and `_` (quote other names with backticks), found `{c}`"
            )
        };
        let bracket = "`[` is reserved: a path only selects struct fields; use `.list` or `.arr` methods for list and array elements".to_string();
        let cases: &[(&str, usize, String)] = &[
            ("", 0, "expected a field name, found end of path".into()),
            ("a.", 2, "expected a field name, found end of path".into()),
            ("a..b", 2, "expected a field name, found `.`".into()),
            (".a", 0, "expected a field name, found `.`".into()),
            ("ä.", 2, "expected a field name, found end of path".into()),
            ("a.0", 2, start("0")),
            ("0", 0, start("0")),
            ("-a", 0, start("-")),
            ("a.[0]", 2, start("[")),
            ("a]", 1, inner("]")),
            ("my col", 2, inner(" ")),
            ("a.b$", 3, inner("$")),
            ("a[0]", 1, bracket.clone()),
            ("a.b[]", 3, bracket.clone()),
            ("a.b[*].c", 3, bracket),
            (
                "a`b`",
                1,
                "expected `.` or end of path, found `` ` ``".into(),
            ),
            ("`a`b", 3, inner("b")),
            ("`ab", 0, "unterminated backtick-quoted name".into()),
            ("a.`b``", 2, "unterminated backtick-quoted name".into()),
            (
                "a.*",
                2,
                "selecting all fields with `.*` is not supported yet".into(),
            ),
        ];
        for (path, pos, msg) in cases {
            let e = parse_field_path(path).unwrap_err();
            assert_eq!(
                (e.pos, e.msg.as_str()),
                (*pos, msg.as_str()),
                "unexpected error for {path:?}"
            );
        }
    }

    #[test]
    fn test_field_path_names() {
        let names = |s: &str| {
            let (out, root) = field_path_names(&s.into());
            (out.to_string(), root.to_string())
        };
        assert_eq!(names("a.b.c"), ("c".into(), "a".into()));
        assert_eq!(names("a"), ("a".into(), "a".into()));
        assert_eq!(names("`x.y`"), ("x.y".into(), "x.y".into()));
        assert_eq!(names("a[b"), ("a[b".into(), "a[b".into()));
    }

    #[test]
    fn test_write_path_roundtrip() {
        for path in ["a.b.c", "`x.y`.z", "`a``b`", "a.``", "_a.`0`.`my col`.b_1"] {
            let mut out = String::new();
            for (i, name) in parse_field_path(path).unwrap().iter().enumerate() {
                if i > 0 {
                    out.push('.');
                }
                write_name(&mut out, name);
            }
            assert_eq!(out, path);
        }
    }
}
