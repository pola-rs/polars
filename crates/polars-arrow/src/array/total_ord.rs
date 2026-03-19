use polars_utils::total_ord::TotalEq;

use crate::array::Array;

impl TotalEq for Box<dyn Array> {
    fn tot_eq(&self, other: &Self) -> bool {
        self == other
    }
}
