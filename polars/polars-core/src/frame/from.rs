use arrow::array::StructArray;

use crate::prelude::*;

impl TryFrom<StructArray> for DataFrame {
    type Error = PolarsError;

    fn try_from(arr: StructArray) -> PolarsResult<Self> {
        let (fld, arrs, nulls) = arr.into_data();
        if nulls.is_some() {
            return Err(PolarsError::ComputeError(
                "cannot deserialize struct with nulls into a DataFrame".into(),
            ));
        }
        let columns = fld
            .iter()
            .zip(arrs)
            .map(|(fld, arr)| {
                // Safety
                // reported data type is correct
                unsafe { Series::try_from_arrow_unchecked(&fld.name, vec![arr], fld.data_type()) }
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        DataFrame::new(columns)
    }
}
