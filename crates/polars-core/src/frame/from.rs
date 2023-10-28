use arrow::array::StructArray;

use crate::prelude::*;

impl TryFrom<StructArray> for DataFrame {
    type Error = PolarsError;

    fn try_from(arr: StructArray) -> PolarsResult<Self> {
        let (fld, arrs, nulls) = arr.into_data();
        polars_ensure!(
            nulls.is_none(),
            ComputeError: "cannot deserialize struct with nulls into a dataframe"
        );
        let columns = fld
            .iter()
            .zip(arrs)
            .map(|(fld, arr)| {
                // Safety
                // reported data type is correct
                unsafe { Series::_try_from_arrow_unchecked(&fld.name, vec![arr], fld.data_type()) }
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

impl From<&ArrowSchema> for DataFrame {
    fn from(schema: &ArrowSchema) -> Self {
        let cols = schema
            .fields
            .iter()
            .map(|fld| Series::new_empty(fld.name.as_str(), &(fld.data_type().into())))
            .collect();
        DataFrame::new_no_checks(cols)
    }
}
