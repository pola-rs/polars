use super::*;

pub(super) fn concat_expr(s: &[Column], rechunk: bool) -> PolarsResult<Column> {
    let mut first = s[0].clone();

    for s in &s[1..] {
        first.append(s)?;
    }
    if rechunk {
        first = first.rechunk()
    }
    Ok(first)
}
