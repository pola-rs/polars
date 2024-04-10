use std::fmt::{Display, Formatter};

use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::dsl::SpecialEq;
use crate::map_as_slice;
use crate::prelude::SeriesUdf;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum BusinessFunction {
    #[cfg(feature = "business")]
    BusinessDayCount {
        week_mask: [bool; 7],
        holidays: Vec<i32>,
    },
}

impl Display for BusinessFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use BusinessFunction::*;
        let s = match self {
            #[cfg(feature = "business")]
            &BusinessDayCount { .. } => "business_day_count",
        };
        write!(f, "{s}")
    }
}
impl From<BusinessFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: BusinessFunction) -> Self {
        use BusinessFunction::*;
        match func {
            #[cfg(feature = "business")]
            BusinessDayCount {
                week_mask,
                holidays,
            } => {
                map_as_slice!(business_day_count, week_mask, &holidays)
            },
        }
    }
}

#[cfg(feature = "business")]
pub(super) fn business_day_count(
    s: &[Series],
    week_mask: [bool; 7],
    holidays: &[i32],
) -> PolarsResult<Series> {
    let start = &s[0];
    let end = &s[1];
    polars_ops::prelude::business_day_count(start, end, week_mask, holidays)
}
