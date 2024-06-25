use polars_parquet::parquet::error::{ParquetError, ParquetResult};
use polars_parquet::parquet::types::{decode, NativeType};

#[derive(Debug)]
pub struct PrimitivePageDict<T: NativeType> {
    values: Vec<T>,
}

impl<T: NativeType> PrimitivePageDict<T> {
    pub fn new(values: Vec<T>) -> Self {
        Self { values }
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }

    #[inline]
    pub fn value(&self, index: usize) -> ParquetResult<&T> {
        self.values.get(index).ok_or_else(|| {
            ParquetError::OutOfSpec(
                "The data page has an index larger than the dictionary page values".to_string(),
            )
        })
    }
}

pub fn read<T: NativeType>(
    buf: &[u8],
    num_values: usize,
    _is_sorted: bool,
) -> ParquetResult<PrimitivePageDict<T>> {
    let size_of = std::mem::size_of::<T>();

    let typed_size = num_values.wrapping_mul(size_of);

    let values = buf.get(..typed_size).ok_or_else(|| {
        ParquetError::OutOfSpec(
            "The number of values declared in the dict page does not match the length of the page"
                .to_string(),
        )
    })?;

    let values = values.chunks_exact(size_of).map(decode::<T>).collect();

    Ok(PrimitivePageDict::new(values))
}
