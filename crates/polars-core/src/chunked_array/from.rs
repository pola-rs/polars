use super::*;

/// Imports Arrow arrays as the chunks of a [`ChunkedArray`], which is `O(1)` for each of them.
pub fn import_arrow_chunks(chunks: Vec<ArrayRef>) -> Vec<PlArrayRef> {
    chunks
        .iter()
        .map(|chunk| polars_array::arrow::import::from_arrow(&**chunk))
        .collect()
}

impl<T, A> From<A> for ChunkedArray<T>
where
    T: PolarsDataType<Array = A>,
    A: StaticArray,
{
    fn from(arr: A) -> Self {
        Self::with_chunk(PlSmallStr::EMPTY, arr)
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsDataType,
{
    pub fn with_chunk<A>(name: PlSmallStr, arr: A) -> Self
    where
        A: StaticArray,
        T: PolarsDataType<Array = A>,
    {
        unsafe { Self::from_chunks(name, vec![arr.into_boxed()]) }
    }

    pub fn with_chunk_like<A>(ca: &Self, arr: A) -> Self
    where
        A: StaticArray,
        T: PolarsDataType<Array = A>,
    {
        Self::from_chunk_iter_like(ca, std::iter::once(arr))
    }

    pub fn from_chunk_iter<I>(name: PlSmallStr, iter: I) -> Self
    where
        I: IntoIterator,
        T: PolarsDataType<Array = <I as IntoIterator>::Item>,
        <I as IntoIterator>::Item: StaticArray,
    {
        let chunks = iter.into_iter().map(StaticArray::into_boxed).collect();
        unsafe { Self::from_chunks(name, chunks) }
    }

    pub fn from_chunk_iter_like<I>(ca: &Self, iter: I) -> Self
    where
        I: IntoIterator,
        T: PolarsDataType<Array = <I as IntoIterator>::Item>,
        <I as IntoIterator>::Item: StaticArray,
    {
        let chunks = iter.into_iter().map(StaticArray::into_boxed).collect();
        unsafe {
            Self::from_chunks_and_dtype_unchecked(ca.name().clone(), chunks, ca.dtype().clone())
        }
    }

    pub fn try_from_chunk_iter<I, A, E>(name: PlSmallStr, iter: I) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<A, E>>,
        T: PolarsDataType<Array = A>,
        A: StaticArray,
    {
        let chunks: Result<_, _> = iter.into_iter().map(|x| Ok(x?.into_boxed())).collect();
        unsafe { Ok(Self::from_chunks(name, chunks?)) }
    }

    pub(crate) fn from_chunk_iter_and_field<I>(field: Arc<Field>, chunks: I) -> Self
    where
        I: IntoIterator,
        T: PolarsDataType<Array = <I as IntoIterator>::Item>,
        <I as IntoIterator>::Item: StaticArray,
    {
        assert_eq!(
            std::mem::discriminant(&T::get_static_dtype()),
            std::mem::discriminant(&field.dtype)
        );

        let mut length = 0;
        let mut null_count = 0;
        let chunks = chunks
            .into_iter()
            .map(|x| {
                length += x.len();
                null_count += x.null_count();
                x.into_boxed()
            })
            .collect();

        unsafe { ChunkedArray::new_with_dims(field, chunks, length, null_count) }
    }

    /// Creates a [`ChunkedArray`] from Arrow chunks, importing each one.
    ///
    /// Importing hands the backing buffers over rather than copying the elements, so this is
    /// `O(1)` per chunk — see [`polars_array::arrow::import`]. It is what the boundaries where
    /// data arrives as Arrow use: an I/O reader, an FFI import, a kernel of `polars-compute`.
    ///
    /// # Safety
    /// The physical type of all chunks must match the [`PolarsDataType`] `T`.
    pub unsafe fn from_arrow_chunks(name: PlSmallStr, chunks: Vec<ArrayRef>) -> Self {
        unsafe { Self::from_chunks(name, import_arrow_chunks(chunks)) }
    }

    /// Creates a [`ChunkedArray`] of `dtype` from Arrow chunks, importing each one.
    ///
    /// This is [`ChunkedArray::from_arrow_chunks`] for a type whose [`DataType`] the chunks do not
    /// imply — a nested one, whose inner type the imported chunks no longer carry.
    ///
    /// # Safety
    /// The physical type of all chunks must match `dtype`.
    pub unsafe fn from_arrow_chunks_and_dtype_unchecked(
        name: PlSmallStr,
        chunks: Vec<ArrayRef>,
        dtype: DataType,
    ) -> Self {
        unsafe { Self::from_chunks_and_dtype_unchecked(name, import_arrow_chunks(chunks), dtype) }
    }

    /// Create a new [`ChunkedArray`] from existing chunks.
    ///
    /// The [`DataType`] is the static one of `T`, which for a nested type names no inner type:
    /// the chunks carry no logical type to recover one from, so a nested [`ChunkedArray`] is built
    /// with [`ChunkedArray::from_chunks_and_dtype`] instead.
    ///
    /// # Safety
    /// The physical type of all chunks must match the [`PolarsDataType`] `T`.
    pub unsafe fn from_chunks(name: PlSmallStr, chunks: Vec<PlArrayRef>) -> Self {
        Self::from_chunks_and_dtype(name, chunks, T::get_static_dtype())
    }

    /// # Safety
    /// The Arrow datatype of all chunks must match the [`PolarsDataType`] `T`.
    pub unsafe fn with_chunks(&self, chunks: Vec<PlArrayRef>) -> Self {
        ChunkedArray::new_with_compute_len(self.field.clone(), chunks)
    }

    /// Create a new [`ChunkedArray`] from existing chunks.
    ///
    /// # Safety
    ///
    /// The Arrow datatype of all chunks must match the [`PolarsDataType`] `T`.
    pub unsafe fn from_chunks_and_dtype(
        name: PlSmallStr,
        chunks: Vec<PlArrayRef>,
        dtype: DataType,
    ) -> Self {
        // Assertions in debug mode that check the chunks are laid out the way the dtype says.
        #[cfg(debug_assertions)]
        {
            if let Some(chunk) = chunks.first().filter(|chunk| !chunk.is_empty()) {
                debug_assert_eq!(
                    chunk.array_type(),
                    dtype.to_physical().to_arrow(CompatLevel::newest()).into(),
                );
            }
        }

        Self::from_chunks_and_dtype_unchecked(name, chunks, dtype)
    }

    /// Create a new [`ChunkedArray`] from existing chunks.
    ///
    /// # Safety
    ///
    /// The Arrow datatype of all chunks must match the [`PolarsDataType`] `T`.
    pub(crate) unsafe fn from_chunks_and_dtype_unchecked(
        name: PlSmallStr,
        chunks: Vec<PlArrayRef>,
        dtype: DataType,
    ) -> Self {
        let field = Arc::new(Field::new(name, dtype));
        ChunkedArray::new_with_compute_len(field, chunks)
    }

    /// A [`ChunkedArray`] of `length` nulls, laid out like `ca`.
    ///
    /// The nulls keep the [`scalar`](polars_array::broadcast) representation, so this is `O(1)` in
    /// memory for every type but a struct, which repeats one null per field.
    pub fn full_null_like(ca: &Self, length: usize) -> Self {
        let prototype = ca.chunks.first().expect("a ChunkedArray has a chunk");
        let chunks = vec![polars_array::builder::full_null_like(&**prototype, length)];
        unsafe {
            let mut out =
                Self::from_chunks_and_dtype_unchecked(ca.name().clone(), chunks, ca.dtype().clone());
            out.length = length;
            out.null_count = length;
            out
        }
    }
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    /// Create a new ChunkedArray by taking ownership of the Vec. This operation is zero copy.
    pub fn from_vec(name: PlSmallStr, v: Vec<T::Native>) -> Self {
        Self::with_chunk(name, to_primitive::<T>(v, None))
    }

    /// Create a new ChunkedArray from a Vec and a validity mask.
    pub fn from_vec_validity(
        name: PlSmallStr,
        values: Vec<T::Native>,
        buffer: Option<Bitmap>,
    ) -> Self {
        let arr = to_array::<T>(values, buffer);
        ChunkedArray::new_with_compute_len(
            Arc::new(Field::new(name, T::get_static_dtype())),
            vec![arr],
        )
    }

    /// Create a temporary [`ChunkedArray`] from a slice.
    ///
    /// # Safety
    /// The lifetime will be bound to the lifetime of the slice.
    /// This will not be checked by the borrowchecker.
    pub unsafe fn mmap_slice(name: PlSmallStr, values: &[T::Native]) -> Self {
        Self::with_chunk(
            name,
            polars_array::arrow::import::primitive_from_arrow(&arrow::ffi::mmap::slice(values)),
        )
    }
}

impl BooleanChunked {
    /// Create a temporary [`ChunkedArray`] from a slice.
    ///
    /// # Safety
    /// The lifetime will be bound to the lifetime of the slice.
    /// This will not be checked by the borrowchecker.
    pub unsafe fn mmap_slice(name: PlSmallStr, values: &[u8], offset: usize, len: usize) -> Self {
        let arr = arrow::ffi::mmap::bitmap(values, offset, len).unwrap();
        Self::with_chunk(name, polars_array::arrow::import::boolean_from_arrow(&arr))
    }

    pub fn from_bitmap(name: PlSmallStr, bitmap: Bitmap) -> Self {
        Self::with_chunk(name, PlBooleanArray::from_values(bitmap))
    }
}

impl<'a, T> From<&'a ChunkedArray<T>> for Vec<Option<T::Physical<'a>>>
where
    T: PolarsDataType,
{
    fn from(ca: &'a ChunkedArray<T>) -> Self {
        let mut out = Vec::with_capacity(ca.len());
        for arr in ca.downcast_iter() {
            out.extend(arr.iter())
        }
        out
    }
}
impl From<StringChunked> for Vec<Option<String>> {
    fn from(ca: StringChunked) -> Self {
        ca.iter().map(|opt| opt.map(|s| s.to_string())).collect()
    }
}

impl From<BooleanChunked> for Vec<Option<bool>> {
    fn from(ca: BooleanChunked) -> Self {
        let mut out = Vec::with_capacity(ca.len());
        for arr in ca.downcast_iter() {
            out.extend(arr.iter())
        }
        out
    }
}
