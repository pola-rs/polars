use std::sync::Arc;

use polars_core::error::{PolarsResult, polars_ensure};
use polars_core::prelude::Column;
use polars_ops::series::Roll;
use polars_plan::dsl::{ColumnsUdf, SpecialEq};
use polars_plan::plans::IRBusinessFunction;

pub fn function_expr_to_udf(func: IRBusinessFunction) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    use IRBusinessFunction::*;
    match func {
        BusinessDayCount { week_mask } => {
            map_as_slice!(business_day_count, week_mask)
        },
        AddBusinessDay { week_mask, roll } => {
            map_as_slice!(add_business_days, week_mask, roll)
        },
        IsBusinessDay { week_mask } => {
            map_as_slice!(is_business_day, week_mask)
        },
    }
}

pub(super) fn business_day_count(s: &[Column], week_mask: [bool; 7]) -> PolarsResult<Column> {
    let start = &s[0];
    let end = &s[1];
    let holidays = &s[2];
    polars_ensure!(
        holidays.len() == 1 || start.len() == 1 || end.len() == 1 || holidays.len() == start.len(),
        ShapeMismatch: "number of holiday lists must be either 1 or the number of dates"
    );
    polars_ops::prelude::business_day_count(
        start.as_materialized_series(),
        end.as_materialized_series(),
        week_mask,
        holidays.as_materialized_series(),
    )
    .map(Column::from)
}

pub(super) fn add_business_days(
    s: &[Column],
    week_mask: [bool; 7],
    roll: Roll,
) -> PolarsResult<Column> {
    let start = &s[0];
    let n = &s[1];
    let holidays = &s[2];
    polars_ensure!(
        holidays.len() == 1 || holidays.len() == start.len(),
        ShapeMismatch: "number of holiday lists must be either 1 or the number of dates"
    );
    polars_ops::prelude::add_business_days(
        start.as_materialized_series(),
        n.as_materialized_series(),
        week_mask,
        holidays.as_materialized_series(),
        roll,
    )
    .map(Column::from)
}

pub(super) fn is_business_day(s: &[Column], week_mask: [bool; 7]) -> PolarsResult<Column> {
    let dates = &s[0];
    let holidays = &s[1];
    polars_ensure!(
        holidays.len() == 1 || holidays.len() == dates.len(),
        ShapeMismatch: "number of holiday lists must be either 1 or the number of dates"
    );
    polars_ops::prelude::is_business_day(
        dates.as_materialized_series(),
        week_mask,
        holidays.as_materialized_series(),
    )
    .map(Column::from)
}
