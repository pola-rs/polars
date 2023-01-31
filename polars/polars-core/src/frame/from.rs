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

impl From<&Schema> for DataFrame {
    fn from(schema: &Schema) -> Self {
        let cols = schema
            .iter()
            .map(|(name, dtype)| Series::new_empty(name, dtype))
            .collect();
        DataFrame::new_no_checks(cols)
    }
}
