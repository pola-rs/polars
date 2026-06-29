use super::*;

#[cfg(all(feature = "concat_str", feature = "strings"))]
/// Horizontally concat string columns in linear time
pub fn concat_str<E: AsRef<[Expr]>>(s: E, separator: &str, ignore_nulls: bool) -> Expr {
    let input = s.as_ref().to_vec();
    let separator = separator.into();

    Expr::Function {
        input,
        function: StringFunction::ConcatHorizontal {
            delimiter: separator,
            ignore_nulls,
        }
        .into(),
    }
}

#[cfg(all(feature = "concat_str", feature = "strings"))]
/// Format the results of an array of expressions using a format string
pub fn format_str(format: &str, args: &[Expr]) -> PolarsResult<Expr> {
    let mut positional_input = std::collections::VecDeque::from_iter(args);
    let mut input = Vec::new();
    let mut automatic_used = false;
    let mut index_used = false;

    // Parse format string into constant literal plus insertion locations per
    // input expression.
    let bytes = format.as_bytes();
    let mut s = Vec::with_capacity(format.len());
    let mut insertions = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < format.len() {
        match bytes[i] {
            b'{' => {
                let escaped = bytes.get(i + 1) == Some(&b'{');
                if escaped {
                    s.push(b'{');
                    i += 2;
                    continue;
                }

                let Some(close_offset) = bytes[i..].iter().position(|b| *b == b'}') else {
                    polars_bail!(InvalidOperation: "unmatched '{{' in format string:\n{format}\n\nYou can escape '{{' by writing '{{{{'.");
                };

                let col_name = &bytes[i + 1..i + close_offset];
                let col_name_str = std::str::from_utf8(col_name).unwrap();
                if col_name.is_empty() {
                    polars_ensure!(!index_used, InvalidOperation: "cannot switch from manual specification to automatic field numbering");
                    let Some(expr) = positional_input.pop_front() else {
                        polars_bail!(ShapeMismatch: "too few arguments given for format string:\n{format}");
                    };
                    input.push(expr.clone());
                    automatic_used = true;
                } else if col_name.iter().all(|b| b.is_ascii_digit()) {
                    polars_ensure!(!automatic_used, InvalidOperation: "cannot switch from automatic field numbering to manual specification");
                    let Some(index) = col_name_str
                        .parse::<usize>()
                        .ok()
                        .filter(|idx| *idx < positional_input.len())
                    else {
                        polars_bail!(InvalidOperation: "out of bounds argument index in format string:\n{format}");
                    };
                    input.push(positional_input[index].clone());
                    index_used = true;
                } else {
                    // [a-zA-Z_][a-zA-Z0-9_]*
                    let valid_col_name = (col_name[0].is_ascii_alphabetic() || col_name[0] == b'_')
                        && col_name[1..]
                            .iter()
                            .all(|b| b.is_ascii_alphanumeric() || *b == b'_');
                    polars_ensure!(valid_col_name, InvalidOperation: "unacceptable column name '{col_name_str}' in format string, must be ASCII identifier\n{format}");
                    input.push(Expr::Column(PlSmallStr::from_str(col_name_str)))
                }
                insertions.push(s.len());
                i += close_offset + 1;
            },
            b'}' => {
                let escaped = bytes.get(i + 1) == Some(&b'}');
                polars_ensure!(escaped, InvalidOperation: "unmatched '}}' in format string:\n{format}\n\nYou can escape '}}' by writing '}}}}'.");
                s.push(b'}');
                i += 2;
            },
            _ => {
                s.push(bytes[i]);
                i += 1;
            },
        }
    }

    polars_ensure!(
        index_used || positional_input.is_empty(),
        ShapeMismatch: "number of automatic placeholders should equal the number of arguments"
    );

    Ok(Expr::Function {
        input,
        function: StringFunction::Format {
            format: PlSmallStr::from(std::str::from_utf8(&s).unwrap()),
            insertions: insertions.into(),
        }
        .into(),
    })
}

/// Concat lists entries.
pub fn concat_list<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(s: E) -> PolarsResult<Expr> {
    let s: Vec<_> = s.as_ref().iter().map(|e| e.clone().into()).collect();

    polars_ensure!(!s.is_empty(), ComputeError: "`concat_list` needs one or more expressions");

    Ok(Expr::Function {
        input: s,
        function: FunctionExpr::ListExpr(ListFunction::Concat),
    })
}

/// Horizontally concatenate columns into a single array-type column.
pub fn concat_arr(input: Vec<Expr>) -> PolarsResult<Expr> {
    feature_gated!("dtype-array", {
        polars_ensure!(!input.is_empty(), ComputeError: "`concat_arr` needs one or more expressions");

        Ok(Expr::Function {
            input,
            function: FunctionExpr::ArrayExpr(ArrayFunction::Concat),
        })
    })
}

pub fn concat_expr<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(
    s: E,
    rechunk: bool,
) -> PolarsResult<Expr> {
    let s: Vec<_> = s.as_ref().iter().map(|e| e.clone().into()).collect();
    polars_ensure!(!s.is_empty(), ComputeError: "`concat_expr` needs one or more expressions");
    Ok(Expr::n_ary(FunctionExpr::ConcatExpr(rechunk), s))
}
