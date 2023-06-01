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
        // change the list-value array to the keys and store the dictionary values in the datatype.
        // if a global string cache is set, we also must modify the keys.
        DataType::List(inner) if *inner == DataType::Categorical(None) => {
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
            let new_array = ListArray::new(
                arrow_dtype,
                list_arr.offsets().clone(),
                cat.array_ref(0).clone(),
                list_arr.validity().cloned(),
            );
            chunks.clear();
            chunks.push(Box::new(new_array));
            DataType::List(Box::new(cat.dtype().clone()))
        }
        #[cfg(all(feature = "dtype-array", feature = "dtype-categorical"))]
        DataType::Array(inner, width) if *inner == DataType::Categorical(None) => {
            let array = concatenate_owned_unchecked(chunks).unwrap();
            let list_arr = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
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
            let arrow_dtype = FixedSizeListArray::default_datatype(ArrowDataType::UInt32, width);
            let new_array = FixedSizeListArray::new(
                arrow_dtype,
                cat.array_ref(0).clone(),
                list_arr.validity().cloned(),
            );
            chunks.clear();
            chunks.push(Box::new(new_array));
            DataType::Array(Box::new(cat.dtype().clone()), width)
        }
        _ => dtype,
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
{
    /// Create a new ChunkedArray from existing chunks.
    ///
    /// # Safety
    /// The Arrow datatype of all chunks must match the [`PolarsDataType`] `T`.
    pub unsafe fn from_chunks(name: &str, mut chunks: Vec<ArrayRef>) -> Self {
        let dtype = match T::get_dtype() {
            dtype @ DataType::List(_) => from_chunks_list_dtype(&mut chunks, dtype),
            #[cfg(feature = "dtype-array")]
            dtype @ DataType::Array(_, _) => from_chunks_list_dtype(&mut chunks, dtype),
            dt => dt,
        };
        // assertions in debug mode
        // that check if the data types in the arrays are as expected
        #[cfg(debug_assertions)]
        {
            if !chunks.is_empty() && dtype.is_primitive() {
                assert_eq!(chunks[0].data_type(), &dtype.to_physical().to_arrow())
            }
        }
        let field = Arc::new(Field::new(name, dtype));
        let mut out = ChunkedArray {
            field,
            chunks,
            phantom: PhantomData,
            bit_settings: Default::default(),
            length: 0,
        };
        out.compute_len();
        out
    }

    /// # Safety
    /// The Arrow datatype of all chunks must match the [`PolarsDataType`] `T`.
    pub unsafe fn with_chunks(&self, chunks: Vec<ArrayRef>) -> Self {
        let field = self.field.clone();
        let mut out = ChunkedArray {
            field,
            chunks,
            phantom: PhantomData,
            bit_settings: Default::default(),
            length: 0,
        };
        out.compute_len();
        out
    }
}

impl ListChunked {
    pub(crate) unsafe fn from_chunks_and_dtype_unchecked(
        name: &str,
        chunks: Vec<ArrayRef>,
        dtype: DataType,
    ) -> Self {
        let field = Arc::new(Field::new(name, dtype));
        let mut out = ChunkedArray {
            field,
            chunks,
            phantom: PhantomData,
            bit_settings: Default::default(),
            length: 0,
        };
        out.compute_len();
        out
    }
}

#[cfg(feature = "dtype-array")]
impl ArrayChunked {
    pub(crate) unsafe fn from_chunks_and_dtype_unchecked(
        name: &str,
        chunks: Vec<ArrayRef>,
        dtype: DataType,
    ) -> Self {
        let field = Arc::new(Field::new(name, dtype));
        let mut out = ChunkedArray {
            field,
            chunks,
            phantom: PhantomData,
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
        unsafe { Self::from_chunks(name, vec![arr]) }
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
            ..Default::default()
        };
        out.compute_len();
        out
    }

    /// Create a temporary [`ChunkedArray`] from a slice.
    ///
    /// # Safety
    /// The lifetime will be bound to the lifetime of the slice.
    /// This will not be checked by the borrowchecker.
    pub unsafe fn mmap_slice(name: &str, values: &[T::Native]) -> Self {
        let arr = arrow::ffi::mmap::slice(values);
        Self::from_chunks(name, vec![Box::new(arr)])
    }
}

impl BooleanChunked {
    /// Create a temporary [`ChunkedArray`] from a slice.
    ///
    /// # Safety
    /// The lifetime will be bound to the lifetime of the slice.
    /// This will not be checked by the borrowchecker.
    pub unsafe fn mmap_slice(name: &str, values: &[u8], offset: usize, len: usize) -> Self {
        let arr = arrow::ffi::mmap::bitmap(values, offset, len).unwrap();
        BooleanChunked::from_chunks(name, vec![Box::new(arr)])
    }
}
