use crate::prelude::*;

impl<T> IsIn for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn is_in(&self, list_array: &ListChunked) -> Result<BooleanChunked> {
        match list_array.dtype() {
            DataType::List(dt) if self.dtype() == dt => {
                let ca: BooleanChunked = self
                    .into_iter()
                    .zip(list_array.into_iter())
                    .map(|(value, series)| match (value, series) {
                        (val, Some(series)) => {
                            let ca = series.unpack::<T>().unwrap();
                            ca.into_iter().any(|a| a == val)
                        }
                        _ => false,
                    })
                    .collect();
                Ok(ca)
            }
            _ => Err(PolarsError::DataTypeMisMatch(
                format!(
                    "cannot do is_in operation with left a dtype: {:?} and right a dtype {:?}",
                    self.dtype(),
                    list_array.dtype()
                )
                .into(),
            )),
        }
    }
}
impl IsIn for Utf8Chunked {
    fn is_in(&self, list_array: &ListChunked) -> Result<BooleanChunked> {
        match list_array.dtype() {
            DataType::List(dt) if self.dtype() == dt => {
                let ca: BooleanChunked = self
                    .into_iter()
                    .zip(list_array.into_iter())
                    .map(|(value, series)| match (value, series) {
                        (val, Some(series)) => {
                            let ca = series.unpack::<Utf8Type>().unwrap();
                            ca.into_iter().any(|a| a == val)
                        }
                        _ => false,
                    })
                    .collect();
                Ok(ca)
            }
            _ => Err(PolarsError::DataTypeMisMatch(
                format!(
                    "cannot do is_in operation with left a dtype: {:?} and right a dtype {:?}",
                    self.dtype(),
                    list_array.dtype()
                )
                .into(),
            )),
        }
    }
}

impl IsIn for BooleanChunked {
    fn is_in(&self, list_array: &ListChunked) -> Result<BooleanChunked> {
        match list_array.dtype() {
            DataType::List(dt) if self.dtype() == dt => {
                let ca: BooleanChunked = self
                    .into_iter()
                    .zip(list_array.into_iter())
                    .map(|(value, series)| match (value, series) {
                        (val, Some(series)) => {
                            let ca = series.unpack::<BooleanType>().unwrap();
                            ca.into_iter().any(|a| a == val)
                        }
                        _ => false,
                    })
                    .collect();
                Ok(ca)
            }
            _ => Err(PolarsError::DataTypeMisMatch(
                format!(
                    "cannot do is_in operation with left a dtype: {:?} and right a dtype {:?}",
                    self.dtype(),
                    list_array.dtype()
                )
                .into(),
            )),
        }
    }
}

impl IsIn for CategoricalChunked {
    fn is_in(&self, list_array: &ListChunked) -> Result<BooleanChunked> {
        self.cast::<UInt32Type>().unwrap().is_in(list_array)
    }
}

impl IsIn for ListChunked {}
