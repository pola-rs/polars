use polars_arrow::array::list::AnonymousBuilder;
use polars_arrow::prelude::*;

use super::*;

pub trait ListBuilderTrait {
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => self.append_null(),
        }
    }
    fn append_series(&mut self, s: &Series);
    fn append_null(&mut self);
    fn finish(&mut self) -> ListChunked;
}

impl<S: ?Sized> ListBuilderTrait for Box<S>
where
    S: ListBuilderTrait,
{
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        (**self).append_opt_series(opt_s)
    }

    fn append_series(&mut self, s: &Series) {
        (**self).append_series(s)
    }

    fn append_null(&mut self) {
        (**self).append_null()
    }

    fn finish(&mut self) -> ListChunked {
        (**self).finish()
    }
}

pub struct ListPrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    pub builder: LargePrimitiveBuilder<T::Native>,
    field: Field,
    fast_explode: bool,
}

macro_rules! finish_list_builder {
    ($self:ident) => {{
        let arr = $self.builder.as_box();

        let mut ca = ListChunked {
            field: Arc::new($self.field.clone()),
            chunks: vec![arr],
            phantom: PhantomData,
            ..Default::default()
        };
        ca.compute_len();
        if $self.fast_explode {
            ca.set_fast_explode()
        }
        ca
    }};
}

impl<T> ListPrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    pub fn new(
        name: &str,
        capacity: usize,
        values_capacity: usize,
        logical_type: DataType,
    ) -> Self {
        let values = MutablePrimitiveArray::<T::Native>::with_capacity(values_capacity);
        let builder = LargePrimitiveBuilder::<T::Native>::new_with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(Box::new(logical_type)));

        Self {
            builder,
            field,
            fast_explode: true,
        }
    }

    pub fn append_slice(&mut self, items: &[T::Native]) {
        let values = self.builder.mut_values();
        values.extend_from_slice(items);
        self.builder.try_push_valid().unwrap();

        if items.is_empty() {
            self.fast_explode = false;
        }
    }

    pub fn append_opt_slice(&mut self, opt_v: Option<&[T::Native]>) {
        match opt_v {
            Some(items) => self.append_slice(items),
            None => {
                self.builder.push_null();
            }
        }
    }
    /// Appends from an iterator over values
    #[inline]
    pub fn append_iter_values<I: Iterator<Item = T::Native> + TrustedLen>(&mut self, iter: I) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        // Safety
        // trusted len, trust the type system
        unsafe { values.extend_trusted_len_values_unchecked(iter) };
        self.builder.try_push_valid().unwrap();
    }

    /// Appends from an iterator over values
    #[inline]
    pub fn append_iter<I: Iterator<Item = Option<T::Native>> + TrustedLen>(&mut self, iter: I) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        // Safety
        // trusted len, trust the type system
        unsafe { values.extend_trusted_len_unchecked(iter) };
        self.builder.try_push_valid().unwrap();
    }
}

impl<T> ListBuilderTrait for ListPrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    #[inline]
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => {
                self.append_series(s);
            }
            None => self.append_null(),
        }
    }

    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.push_null();
    }

    #[inline]
    fn append_series(&mut self, s: &Series) {
        if s.is_empty() {
            self.fast_explode = false;
        }
        let physical = s.to_physical_repr();
        let ca = physical.unpack::<T>().unwrap();
        let values = self.builder.mut_values();

        ca.downcast_iter().for_each(|arr| {
            if !arr.has_validity() {
                values.extend_from_slice(arr.values().as_slice())
            } else {
                // Safety:
                // Arrow arrays are trusted length iterators.
                unsafe { values.extend_trusted_len_unchecked(arr.into_iter()) }
            }
        });
        // overflow of i64 is far beyond polars capable lengths.
        unsafe { self.builder.try_push_valid().unwrap_unchecked() };
    }

    fn finish(&mut self) -> ListChunked {
        finish_list_builder!(self)
    }
}

type LargePrimitiveBuilder<T> = MutableListArray<i64, MutablePrimitiveArray<T>>;
type LargeListUtf8Builder = MutableListArray<i64, MutableUtf8Array<i64>>;
#[cfg(feature = "dtype-binary")]
type LargeListBinaryBuilder = MutableListArray<i64, MutableBinaryArray<i64>>;
type LargeListBooleanBuilder = MutableListArray<i64, MutableBooleanArray>;

pub struct ListUtf8ChunkedBuilder {
    builder: LargeListUtf8Builder,
    field: Field,
    fast_explode: bool,
}

impl ListUtf8ChunkedBuilder {
    pub fn new(name: &str, capacity: usize, values_capacity: usize) -> Self {
        let values = MutableUtf8Array::<i64>::with_capacity(values_capacity);
        let builder = LargeListUtf8Builder::new_with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(Box::new(DataType::Utf8)));

        ListUtf8ChunkedBuilder {
            builder,
            field,
            fast_explode: true,
        }
    }

    pub fn append_trusted_len_iter<'a, I: Iterator<Item = Option<&'a str>> + TrustedLen>(
        &mut self,
        iter: I,
    ) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        // Safety
        // trusted len, trust the type system
        unsafe { values.extend_trusted_len_unchecked(iter) };
        self.builder.try_push_valid().unwrap();
    }

    pub fn append_values_iter<'a, I: Iterator<Item = &'a str>>(&mut self, iter: I) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        values.extend_values(iter);
        self.builder.try_push_valid().unwrap();
    }

    pub(crate) fn append(&mut self, ca: &Utf8Chunked) {
        let value_builder = self.builder.mut_values();
        value_builder.try_extend(ca).unwrap();
        self.builder.try_push_valid().unwrap();
    }
}

impl ListBuilderTrait for ListUtf8ChunkedBuilder {
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => {
                self.append_null();
            }
        }
    }

    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.push_null();
    }

    fn append_series(&mut self, s: &Series) {
        if s.is_empty() {
            self.fast_explode = false;
        }
        let ca = s.utf8().unwrap();
        self.append(ca)
    }

    fn finish(&mut self) -> ListChunked {
        finish_list_builder!(self)
    }
}

#[cfg(feature = "dtype-binary")]
pub struct ListBinaryChunkedBuilder {
    builder: LargeListBinaryBuilder,
    field: Field,
    fast_explode: bool,
}

#[cfg(feature = "dtype-binary")]
impl ListBinaryChunkedBuilder {
    pub fn new(name: &str, capacity: usize, values_capacity: usize) -> Self {
        let values = MutableBinaryArray::<i64>::with_capacity(values_capacity);
        let builder = LargeListBinaryBuilder::new_with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(Box::new(DataType::Binary)));

        ListBinaryChunkedBuilder {
            builder,
            field,
            fast_explode: true,
        }
    }

    pub fn append_trusted_len_iter<'a, I: Iterator<Item = Option<&'a [u8]>> + TrustedLen>(
        &mut self,
        iter: I,
    ) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        // Safety
        // trusted len, trust the type system
        unsafe { values.extend_trusted_len_unchecked(iter) };
        self.builder.try_push_valid().unwrap();
    }

    pub fn append_values_iter<'a, I: Iterator<Item = &'a [u8]>>(&mut self, iter: I) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        values.extend_values(iter);
        self.builder.try_push_valid().unwrap();
    }

    pub(crate) fn append(&mut self, ca: &BinaryChunked) {
        let value_builder = self.builder.mut_values();
        value_builder.try_extend(ca).unwrap();
        self.builder.try_push_valid().unwrap();
    }
}

#[cfg(feature = "dtype-binary")]
impl ListBuilderTrait for ListBinaryChunkedBuilder {
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => {
                self.append_null();
            }
        }
    }

    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.push_null();
    }

    fn append_series(&mut self, s: &Series) {
        if s.is_empty() {
            self.fast_explode = false;
        }
        let ca = s.binary().unwrap();
        self.append(ca)
    }

    fn finish(&mut self) -> ListChunked {
        finish_list_builder!(self)
    }
}

pub struct ListBooleanChunkedBuilder {
    builder: LargeListBooleanBuilder,
    field: Field,
    fast_explode: bool,
}

impl ListBooleanChunkedBuilder {
    pub fn new(name: &str, capacity: usize, values_capacity: usize) -> Self {
        let values = MutableBooleanArray::with_capacity(values_capacity);
        let builder = LargeListBooleanBuilder::new_with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(Box::new(DataType::Boolean)));

        Self {
            builder,
            field,
            fast_explode: true,
        }
    }

    #[inline]
    pub fn append_iter<I: Iterator<Item = Option<bool>> + TrustedLen>(&mut self, iter: I) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        // Safety
        // trusted len, trust the type system
        unsafe { values.extend_trusted_len_unchecked(iter) };
        self.builder.try_push_valid().unwrap();
    }

    #[inline]
    pub(crate) fn append(&mut self, ca: &BooleanChunked) {
        if ca.is_empty() {
            self.fast_explode = false;
        }
        let value_builder = self.builder.mut_values();
        value_builder.extend(ca);
        self.builder.try_push_valid().unwrap();
    }
}

impl ListBuilderTrait for ListBooleanChunkedBuilder {
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => {
                self.append_null();
            }
        }
    }

    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.push_null();
    }

    #[inline]
    fn append_series(&mut self, s: &Series) {
        let ca = s.bool().unwrap();
        self.append(ca)
    }

    fn finish(&mut self) -> ListChunked {
        finish_list_builder!(self)
    }
}

pub fn get_list_builder(
    dt: &DataType,
    value_capacity: usize,
    list_capacity: usize,
    name: &str,
) -> PolarsResult<Box<dyn ListBuilderTrait>> {
    let physical_type = dt.to_physical();

    let _err = || -> PolarsResult<Box<dyn ListBuilderTrait>> {
        Err(PolarsError::ComputeError(
            format!(
                "list builder not supported for this dtype: {}",
                &physical_type
            )
            .into(),
        ))
    };

    match &physical_type {
        #[cfg(feature = "object")]
        DataType::Object(_) => _err(),
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(_) => Ok(Box::new(AnonymousOwnedListBuilder::new(
            name,
            list_capacity,
            Some(physical_type),
        ))),
        DataType::List(_) => Ok(Box::new(AnonymousOwnedListBuilder::new(
            name,
            list_capacity,
            Some(physical_type),
        ))),
        _ => {
            macro_rules! get_primitive_builder {
                ($type:ty) => {{
                    let builder = ListPrimitiveChunkedBuilder::<$type>::new(
                        name,
                        list_capacity,
                        value_capacity,
                        dt.clone(),
                    );
                    Box::new(builder)
                }};
            }
            macro_rules! get_bool_builder {
                () => {{
                    let builder =
                        ListBooleanChunkedBuilder::new(&name, list_capacity, value_capacity);
                    Box::new(builder)
                }};
            }
            macro_rules! get_utf8_builder {
                () => {{
                    let builder =
                        ListUtf8ChunkedBuilder::new(&name, list_capacity, 5 * value_capacity);
                    Box::new(builder)
                }};
            }
            #[cfg(feature = "dtype-binary")]
            macro_rules! get_binary_builder {
                () => {{
                    let builder =
                        ListBinaryChunkedBuilder::new(&name, list_capacity, 5 * value_capacity);
                    Box::new(builder)
                }};
            }
            Ok(match_dtype_to_logical_apply_macro!(
                physical_type,
                get_primitive_builder,
                get_utf8_builder,
                get_binary_builder,
                get_bool_builder
            ))
        }
    }
}

pub struct AnonymousListBuilder<'a> {
    name: String,
    builder: AnonymousBuilder<'a>,
    fast_explode: bool,
    pub dtype: Option<DataType>,
}

impl Default for AnonymousListBuilder<'_> {
    fn default() -> Self {
        Self::new("", 0, None)
    }
}

impl<'a> AnonymousListBuilder<'a> {
    pub fn new(name: &str, capacity: usize, inner_dtype: Option<DataType>) -> Self {
        Self {
            name: name.into(),
            builder: AnonymousBuilder::new(capacity),
            fast_explode: true,
            dtype: inner_dtype,
        }
    }

    pub fn append_opt_series(&mut self, opt_s: Option<&'a Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => {
                self.append_null();
            }
        }
    }

    pub fn append_opt_array(&mut self, opt_s: Option<&'a dyn Array>) {
        match opt_s {
            Some(s) => self.append_array(s),
            None => {
                self.append_null();
            }
        }
    }

    pub fn append_array(&mut self, arr: &'a dyn Array) {
        self.builder.push(arr)
    }

    #[inline]
    pub fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.push_null();
    }

    #[inline]
    pub fn append_empty(&mut self) {
        self.fast_explode = false;
        self.builder.push_empty()
    }

    pub fn append_series(&mut self, s: &'a Series) {
        // empty arrays tend to be null type and thus differ
        // if we would push it the concat would fail.
        if s.is_empty() && matches!(s.dtype(), DataType::Null) {
            self.append_empty();
        } else {
            match s.dtype() {
                #[cfg(feature = "dtype-struct")]
                DataType::Struct(_) => {
                    let arr = &**s.array_ref(0);
                    self.builder.push(arr)
                }
                _ => {
                    self.builder.push_multiple(s.chunks());
                }
            }
        }
    }

    pub fn finish(&mut self) -> ListChunked {
        // don't use self from here on one
        let slf = std::mem::take(self);
        if slf.builder.is_empty() {
            ListChunked::full_null_with_dtype(&slf.name, 0, &slf.dtype.unwrap_or(DataType::Null))
        } else {
            let dtype = slf.dtype.map(|dt| dt.to_physical().to_arrow());
            let arr = slf.builder.finish(dtype.as_ref()).unwrap();
            let dtype = DataType::from(arr.data_type());
            let mut ca = unsafe { ListChunked::from_chunks("", vec![Box::new(arr)]) };

            if slf.fast_explode {
                ca.set_fast_explode();
            }

            ca.field = Arc::new(Field::new(&slf.name, dtype));
            ca
        }
    }
}

pub struct AnonymousOwnedListBuilder {
    name: String,
    builder: AnonymousBuilder<'static>,
    owned: Vec<Series>,
    inner_dtype: Option<DataType>,
    fast_explode: bool,
}

impl Default for AnonymousOwnedListBuilder {
    fn default() -> Self {
        Self::new("", 0, None)
    }
}

impl ListBuilderTrait for AnonymousOwnedListBuilder {
    fn append_series(&mut self, s: &Series) {
        if s.is_empty() {
            self.append_empty();
        } else {
            // Safety
            // we deref a raw pointer with a lifetime that is not static
            // it is safe because we also clone Series (Arc +=1) and therefore the &dyn Arrays
            // will not be dropped until the owned series are dropped
            unsafe {
                match s.dtype() {
                    #[cfg(feature = "dtype-struct")]
                    DataType::Struct(_) => {
                        self.builder.push(&*(&**s.array_ref(0) as *const dyn Array))
                    }
                    _ => {
                        self.builder
                            .push_multiple(&*(s.chunks().as_ref() as *const [ArrayRef]));
                    }
                }
            }
            // this make sure that the underlying ArrayRef's are not dropped
            self.owned.push(s.clone());
        }
    }

    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        self.builder.push_null()
    }

    fn finish(&mut self) -> ListChunked {
        // don't use self from here on one
        let slf = std::mem::take(self);
        if slf.builder.is_empty() {
            // not really empty, there were empty null list added probably e.g. []
            let real_length = slf.builder.offsets().len() - 1;
            if real_length > 0 {
                let dtype = slf.inner_dtype.unwrap_or(NULL_DTYPE).to_arrow();
                let array = new_null_array(dtype.clone(), real_length);
                let dtype = ListArray::<i64>::default_datatype(dtype);
                let array = ListArray::new(dtype, slf.builder.take_offsets().into(), array, None);
                // safety: same type
                unsafe { ListChunked::from_chunks(&slf.name, vec![Box::new(array)]) }
            } else {
                ListChunked::full_null_with_dtype(
                    &slf.name,
                    0,
                    &slf.inner_dtype.unwrap_or(DataType::Null),
                )
            }
        } else {
            let inner_dtype = slf.inner_dtype.map(|dt| dt.to_physical().to_arrow());
            let arr = slf.builder.finish(inner_dtype.as_ref()).unwrap();
            let dtype = DataType::from(arr.data_type());
            // safety: same type
            let mut ca = unsafe { ListChunked::from_chunks("", vec![Box::new(arr)]) };

            if slf.fast_explode {
                ca.set_fast_explode();
            }

            ca.field = Arc::new(Field::new(&slf.name, dtype));
            ca
        }
    }
}

impl AnonymousOwnedListBuilder {
    pub fn new(name: &str, capacity: usize, inner_dtype: Option<DataType>) -> Self {
        Self {
            name: name.into(),
            builder: AnonymousBuilder::new(capacity),
            owned: Vec::with_capacity(capacity),
            inner_dtype,
            fast_explode: true,
        }
    }

    #[inline]
    pub fn append_empty(&mut self) {
        self.fast_explode = false;
        self.builder.push_empty()
    }
}
