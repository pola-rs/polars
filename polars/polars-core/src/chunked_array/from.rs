use super::*;

fn from_chunks_default<T: PolarsDataType>(name: &str, chunks: Vec<ArrayRef>) -> ChunkedArray<T> {
    let field = Arc::new(Field::new(name, datatype));
    let mut out = ChunkedArray {
        field,
        chunks,
        phantom: PhantomData,
        categorical_map: None,
        bit_settings: Default::default(),
        length: 0,
    };
    out.compute_len();
    out

}

impl<T> ChunkedArray<T>
    where
        T: PolarsDataType,
{
    /// Create a new ChunkedArray from existing chunks.
    pub fn from_chunks(name: &str, mut chunks: Vec<ArrayRef>) -> Self {
        // TODO! make this function accept a dtype and make it unsafe
        // prevent List<Null> if the inner list type is known.
        let datatype = if matches!(T::get_dtype(), DataType::List(_)) {
            let dtype = if let Some(arr) = chunks.get(0) {
                arr.data_type().into()
            } else {
                T::get_dtype()
            };

            match dtype {
                #[cfg(feature = "dtype-categorical")]
                DataType::List(inner) if *inner == DataType::Categorical(None) => {
                    let array = concatenate_owned_unchecked(&chunks).unwrap();
                    let list_arr = array.as_any().downcast_ref::<ListArray<i64>>().unwrap();
                    let values_arr = list_arr.values();
                    let cat = Series::try_from(("", values_arr.clone())).unwrap();

                    // we nest only the physical representation
                    // the mapping is still in our rev-map
                    let arrow_dtype =
                        ListArray::<i64>::default_datatype(ArrowDataType::UInt32);
                    let new_array = unsafe {
                        ListArray::new_unchecked(
                            arrow_dtype,
                            list_arr.offsets().clone(),
                            cat.array_ref(0).clone(),
                            list_arr.validity().cloned(),
                        )
                    };
                    chunks.clear();
                    chunks.push(Box::new(new_array));
                    DataType::List(Box::new(cat.dtype().clone()))
                }
                _ => dtype,
            }
        } else {
            T::get_dtype()
        };
        let field = Arc::new(Field::new(name, datatype));
        let mut out = ChunkedArray {
            field,
            chunks,
            phantom: PhantomData,
            categorical_map: None,
            bit_settings: Default::default(),
            length: 0,
        };
        out.compute_len();
        out
    }
}

// A hack to save compiler bloat for null arrays
impl Int32Chunked {
    pub(crate) fn new_null(name: &str, len: usize) -> Self {
        let arr = arrow::array::new_null_array(ArrowDataType::Null, len);
        let field = Arc::new(Field::new(name, DataType::Null));
        let chunks = vec![arr as ArrayRef];
        let mut out = ChunkedArray {
            field,
            chunks,
            phantom: PhantomData,
            categorical_map: None,
            bit_settings: Default::default(),
            length: 0,
        };
        out.compute_len();
        out
    }
}

impl<T> ChunkedArray<T>
    where
        T: PolarsNumericType,
{
    /// Create a new ChunkedArray by taking ownership of the Vec. This operation is zero copy.
    pub fn from_vec(name: &str, v: Vec<T::Native>) -> Self {
        let arr = to_array::<T>(v, None);
        Self::from_chunks(name, vec![arr])
    }

    /// Nullify values in slice with an existing null bitmap
    pub fn new_from_owned_with_null_bitmap(
        name: &str,
        values: Vec<T::Native>,
        buffer: Option<Bitmap>,
    ) -> Self {
        let arr = to_array::<T>(values, buffer);
        let mut out = ChunkedArray {
            field: Arc::new(Field::new(name, T::get_dtype())),
            chunks: vec![arr],
            phantom: PhantomData,
            categorical_map: None,
            ..Default::default()
        };
        out.compute_len();
        out
    }
}

