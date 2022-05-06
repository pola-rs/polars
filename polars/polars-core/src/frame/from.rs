use crate::prelude::*;
use arrow::array::StructArray;

impl TryFrom<StructArray> for DataFrame {
    type Error = PolarsError;

    fn try_from(arr: StructArray) -> Result<Self> {
        let (fld, arrs, nulls) = arr.into_data();
        if nulls.is_some() {
            return Err(PolarsError::ComputeError(
                "cannot deserialze struct with nulls into a DataFrame".into(),
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
            .collect::<Result<Vec<_>>>()?;
        DataFrame::new(columns)
    }
}
