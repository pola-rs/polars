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
pub fn format_str<E: AsRef<[Expr]>>(format: &str, args: E) -> PolarsResult<Expr> {
    let input = args.as_ref().to_vec();

    let mut s = String::with_capacity(format.len());
    let mut insertions = Vec::with_capacity(input.len());
    let mut offset = 0;
    while let Some(j) = format[offset..].find("{}") {
        s.push_str(&format[offset..][..j]);
        insertions.push(s.len());
        offset += j + 2;
    }
    s.push_str(&format[offset..]);

    polars_ensure!(
        insertions.len() == input.len(),
        ShapeMismatch: "number of placeholders should equal the number of arguments"
    );

    Ok(Expr::Function {
        input,
        function: StringFunction::Format {
            format: s.into(),
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
