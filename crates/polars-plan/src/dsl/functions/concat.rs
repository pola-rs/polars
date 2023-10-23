use super::*;

#[cfg(all(feature = "concat_str", feature = "strings"))]
/// Horizontally concat string columns in linear time
pub fn concat_str<E: AsRef<[Expr]>>(s: E, separator: &str) -> Expr {
    let input = s.as_ref().to_vec();
    let separator = separator.to_string();

    Expr::Function {
        input,
        function: StringFunction::ConcatHorizontal(separator).into(),
        options: FunctionOptions {
            collect_groups: ApplyOptions::ElementWise,
            input_wildcard_expansion: true,
            returns_scalar: true,
            ..Default::default()
        },
    }
}

#[cfg(all(feature = "concat_str", feature = "strings"))]
/// Format the results of an array of expressions using a format string
pub fn format_str<E: AsRef<[Expr]>>(format: &str, args: E) -> PolarsResult<Expr> {
    let mut args: std::collections::VecDeque<Expr> = args.as_ref().to_vec().into();

    // Parse the format string, and separate substrings between placeholders
    let segments: Vec<&str> = format.split("{}").collect();

    polars_ensure!(
        segments.len() - 1 == args.len(),
        ShapeMismatch: "number of placeholders should equal the number of arguments"
    );

    let mut exprs: Vec<Expr> = Vec::new();

    for (i, s) in segments.iter().enumerate() {
        if i > 0 {
            if let Some(arg) = args.pop_front() {
                exprs.push(arg);
            }
        }

        if !s.is_empty() {
            exprs.push(lit(s.to_string()))
        }
    }

    Ok(concat_str(exprs, ""))
}

/// Concat lists entries.
pub fn concat_list<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(s: E) -> PolarsResult<Expr> {
    let s: Vec<_> = s.as_ref().iter().map(|e| e.clone().into()).collect();

    polars_ensure!(!s.is_empty(), ComputeError: "`concat_list` needs one or more expressions");

    Ok(Expr::Function {
        input: s,
        function: FunctionExpr::ListExpr(ListFunction::Concat),
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            input_wildcard_expansion: true,
            ..Default::default()
        },
    })
}

pub fn concat_expr<E: AsRef<[IE]>, IE: Into<Expr> + Clone>(
    s: E,
    rechunk: bool,
) -> PolarsResult<Expr> {
    let s: Vec<_> = s.as_ref().iter().map(|e| e.clone().into()).collect();
    polars_ensure!(!s.is_empty(), ComputeError: "`concat_expr` needs one or more expressions");

    Ok(Expr::Function {
        input: s,
        function: FunctionExpr::ConcatExpr(rechunk),
        options: FunctionOptions {
            collect_groups: ApplyOptions::GroupWise,
            input_wildcard_expansion: true,
            cast_to_supertypes: true,
            ..Default::default()
        },
    })
}
