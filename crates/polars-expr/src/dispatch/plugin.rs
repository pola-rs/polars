use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::prelude::{Column, IntoColumn};
use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_plan::dsl::v2::{StatefulUdf, UdfV2Flags};

pub fn call(s: &mut [Column], udf: Arc<StatefulUdf>) -> PolarsResult<Column> {
    let fields = Schema::from_iter(s.iter().map(|c| c.field().into_owned()));
    let series = s
        .iter_mut()
        .map(|c| std::mem::take(c).take_materialized_series())
        .collect::<Vec<_>>();

    let mut state = udf.clone().initialize(&fields)?;

    let flags = udf.flags();
    let insert = state.insert(&series)?;

    assert!(insert.is_none() || flags.contains(UdfV2Flags::INSERT_HAS_OUTPUT));

    if !flags.contains(UdfV2Flags::NEEDS_FINALIZE) || flags.is_elementwise() {
        let field = udf.to_field(&fields)?;
        let out = insert.unwrap_or_else(|| Series::new_empty(field.name, &field.dtype));
        return Ok(out.into_column());
    }

    let finalize = state.finalize()?;
    Ok(match (insert, finalize) {
        (None, None) => {
            let field = udf.to_field(&fields)?;
            Column::new_empty(field.name, &field.dtype)
        },
        (Some(s), None) | (None, Some(s)) => s.into_column(),
        (Some(mut s), Some(s2)) => {
            s.append_owned(s2)?;
            s.into_column()
        },
    })
}
