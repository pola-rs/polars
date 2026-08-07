use std::num::NonZeroU32;

use super::*;

/// Split an ordered window partition into `n` buckets, numbered `1..=n`.
pub fn ntile(n: u32) -> PolarsResult<Expr> {
    let n = NonZeroU32::new(n)
        .ok_or_else(|| polars_err!(InvalidOperation: "`ntile` requires `n` > 0"))?;

    Ok(Expr::n_ary(FunctionExpr::NTile { n }, vec![len()]))
}
