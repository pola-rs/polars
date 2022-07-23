use super::*;

#[allow(clippy::ptr_arg)]
fn from_chunks_list_dtype(chunks: &mut Vec<ArrayRef>, dtype: DataType) -> DataType {
    // ensure we don't get List<null>
    let dtype = if let Some(arr) = chunks.get(0) {
        arr.data_type().into()
    } else {
        dtype
    };

    match dtype {
        #[cfg(feature = "dtype-categorical")]
        // arrow dictionaries are not nested as dictionaries, but only by their keys, so we must
        // change the list-value array to the keys and store the dicitonary values in the datatype.
        // if a global string cache is set, we also must modify the keys.
        DataType::List(inner) if *inner == DataType::Categorical(None) => {
            use polars_arrow::kernels::concatenate::concatenate_owned_unchecked;
            let array = concatenate_owned_unchecked(chunks).unwrap();
            let list_arr = array.as_any().downcast_ref::<ListArray<i64>>().unwrap();
            let values_arr = list_arr.values();
            let cat = unsafe {
                Series::try_from_arrow_unchecked(
                    "",
                    vec![values_arr.clone()],
                    values_arr.data_type(),
                )
                .unwrap()
            };

            // we nest only the physical representation
            // the mapping is still in our rev-map
            let arrow_dtype = ListArray::<i64>::default_datatype(ArrowDataType::UInt32);
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
}

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
{
    /// Create a new ChunkedArray from existing chunks.
    pub fn from_chunks(name: &str, mut chunks: Vec<ArrayRef>) -> Self {
        let dtype = match T::get_dtype() {
            dtype @ DataType::List(_) => from_chunks_list_dtype(&mut chunks, dtype),
            dt => dt,
        };
        let field = Arc::new(Field::new(name, dtype));
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
