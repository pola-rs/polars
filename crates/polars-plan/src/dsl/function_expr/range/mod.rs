mod int_range;

use std::fmt::{Display, Formatter};

use polars_core::prelude::*;
use polars_core::series::Series;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::dsl::SpecialEq;
use crate::map_as_slice;
use crate::prelude::SeriesUdf;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub enum RangeFunction {
    IntRange { step: i64 },
    IntRanges { step: i64 },
}

impl RangeFunction {
    pub(super) fn get_field(&self) -> PolarsResult<Field> {
        use RangeFunction::*;
        let field = match self {
            IntRange { .. } => Field::new("int", DataType::Int64),
            IntRanges { .. } => Field::new("int_range", DataType::List(Box::new(DataType::Int64))),
        };
        Ok(field)
    }
}

impl Display for RangeFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use RangeFunction::*;
        let s = match self {
            IntRange { .. } => "int_range",
            IntRanges { .. } => "int_ranges",
        };
        write!(f, "{s}")
    }
}

impl From<RangeFunction> for SpecialEq<Arc<dyn SeriesUdf>> {
    fn from(func: RangeFunction) -> Self {
        use RangeFunction::*;
        match func {
            IntRange { step } => {
                map_as_slice!(int_range::int_range, step)
            },
            IntRanges { step } => {
                map_as_slice!(int_range::int_ranges, step)
            },
        }
    }
}
