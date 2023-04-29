use polars_arrow::array::fixed_size_list::AnonymousBuilder;
use polars_arrow::array::null::MutableNullArray;
use polars_arrow::prelude::*;

use super::*;

pub trait FixedSizeListBuilderTrait {
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => self.append_null(),
        }
    }
    fn append_series(&mut self, s: &Series);
    fn append_null(&mut self);
    fn finish(&mut self) -> FixedSizeListChunked;
}

impl<S: ?Sized> FixedSizeListBuilderTrait for Box<S>
where
    S: FixedSizeListBuilderTrait,
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

    fn finish(&mut self) -> FixedSizeListChunked {
        (**self).finish()
    }
}

pub struct FixedSizeListPrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    pub builder: FixedSizePrimitiveBuilder<T::Native>,
    field: Field,
    fast_explode: bool,
}

macro_rules! finish_list_builder {
    ($self:ident) => {{
        let arr = $self.builder.as_box();

        let mut ca = FixedSizeListChunked {
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

impl<T> FixedSizeListPrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    pub fn new(name: &str, capacity: usize, logical_type: DataType, inner_size: usize) -> Self {
        let values_capacity = capacity * inner_size;
        let values = MutablePrimitiveArray::<T::Native>::with_capacity(values_capacity);
        let builder = FixedSizePrimitiveBuilder::<T::Native>::new(values, inner_size);
        let field = Field::new(name, DataType::List(Box::new(logical_type)));

        Self {
            builder,
            field,
            fast_explode: true,
        }
    }

    #[inline]
    pub fn append_slice(&mut self, items: &[T::Native]) {
        let values = self.builder.mut_values();
        values.extend_from_slice(items);
        self.builder.try_push_valid().unwrap();

        if items.is_empty() {
            self.fast_explode = false;
        }
    }

    #[inline]
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

impl<T> FixedSizeListBuilderTrait for FixedSizeListPrimitiveChunkedBuilder<T>
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

    fn finish(&mut self) -> FixedSizeListChunked {
        finish_list_builder!(self)
    }
}

type FixedSizePrimitiveBuilder<T> = MutableFixedSizeListArray<MutablePrimitiveArray<T>>;
type FixedSizeListUtf8Builder = MutableFixedSizeListArray<MutableUtf8Array<i64>>;
type FixedSizeListBinaryBuilder = MutableFixedSizeListArray<MutableBinaryArray<i64>>;
type FixedSizeListBooleanBuilder = MutableFixedSizeListArray<MutableBooleanArray>;
type FixedSizeListNullBuilder = MutableFixedSizeListArray<MutableNullArray>;

pub struct FixedSizeListUtf8ChunkedBuilder {
    builder: FixedSizeListUtf8Builder,
    field: Field,
    fast_explode: bool,
}

impl FixedSizeListUtf8ChunkedBuilder {
    pub fn new(name: &str, capacity: usize, inner_size: usize) -> Self {
        let values_capacity = capacity * inner_size;
        let values = MutableUtf8Array::<i64>::with_capacity(values_capacity);
        let builder = FixedSizeListUtf8Builder::new(values, inner_size);
        let field = Field::new(name, DataType::List(Box::new(DataType::Utf8)));

        FixedSizeListUtf8ChunkedBuilder {
            builder,
            field,
            fast_explode: true,
        }
    }

    #[inline]
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

    #[inline]
    pub fn append_values_iter<'a, I: Iterator<Item = &'a str>>(&mut self, iter: I) {
        let values = self.builder.mut_values();

        if iter.size_hint().0 == 0 {
            self.fast_explode = false;
        }
        values.extend_values(iter);
        self.builder.try_push_valid().unwrap();
    }

    #[inline]
    pub(crate) fn append(&mut self, ca: &Utf8Chunked) {
        let value_builder = self.builder.mut_values();
        value_builder.try_extend(ca).unwrap();
        self.builder.try_push_valid().unwrap();
    }
}

impl FixedSizeListBuilderTrait for FixedSizeListUtf8ChunkedBuilder {
    #[inline]
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
        if s.is_empty() {
            self.fast_explode = false;
        }
        let ca = s.utf8().unwrap();
        self.append(ca)
    }

    fn finish(&mut self) -> FixedSizeListChunked {
        finish_list_builder!(self)
    }
}

pub struct FixedSizeListBinaryChunkedBuilder {
    builder: FixedSizeListBinaryBuilder,
    field: Field,
    fast_explode: bool,
}

impl FixedSizeListBinaryChunkedBuilder {
    pub fn new(name: &str, capacity: usize, inner_size: usize) -> Self {
        let values_capacity = capacity * inner_size;
        let values = MutableBinaryArray::<i64>::with_capacity(values_capacity);
        let builder = FixedSizeListBinaryBuilder::new(values, inner_size);
        let field = Field::new(name, DataType::List(Box::new(DataType::Binary)));

        FixedSizeListBinaryChunkedBuilder {
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

impl FixedSizeListBuilderTrait for FixedSizeListBinaryChunkedBuilder {
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

    fn finish(&mut self) -> FixedSizeListChunked {
        finish_list_builder!(self)
    }
}

pub struct FixedSizeListBooleanChunkedBuilder {
    builder: FixedSizeListBooleanBuilder,
    field: Field,
    fast_explode: bool,
}

impl FixedSizeListBooleanChunkedBuilder {
    pub fn new(name: &str, capacity: usize, inner_size: usize) -> Self {
        let values_capacity = capacity * inner_size;
        let values = MutableBooleanArray::with_capacity(values_capacity);
        let builder = FixedSizeListBooleanBuilder::new(values, inner_size);
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

impl FixedSizeListBuilderTrait for FixedSizeListBooleanChunkedBuilder {
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

    fn finish(&mut self) -> FixedSizeListChunked {
        finish_list_builder!(self)
    }
}

impl FixedSizeListBuilderTrait for FixedSizeListNullBuilder {
    #[inline]
    fn append_series(&mut self, _s: &Series) {
        self.push_null()
    }

    #[inline]
    fn append_null(&mut self) {
        self.push_null()
    }

    fn finish(&mut self) -> FixedSizeListChunked {
        unsafe {
            FixedSizeListChunked::from_chunks_and_dtype_unchecked(
                "",
                vec![self.as_box()],
                DataType::List(Box::new(DataType::Null)),
            )
        }
    }
}

pub fn get_fixed_size_list_builder(
    inner_type_logical: &DataType,
    list_capacity: usize,
    name: &str,
    inner_size: usize,
) -> PolarsResult<Box<dyn FixedSizeListBuilderTrait>> {
    let physical_type = inner_type_logical.to_physical();

    match &physical_type {
        #[cfg(feature = "object")]
        DataType::Object(_) => polars_bail!(opq = list_builder, &physical_type),
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(_) => Ok(Box::new(AnonymousOwnedFixedSizeListBuilder::new(
            name,
            list_capacity,
            Some(inner_type_logical.clone()),
            inner_size,
        ))),
        DataType::Null => Ok(Box::new(FixedSizeListNullBuilder::with_capacity(
            list_capacity,
        ))),
        DataType::List(_) => Ok(Box::new(AnonymousOwnedFixedSizeListBuilder::new(
            name,
            list_capacity,
            Some(inner_type_logical.clone()),
            inner_size,
        ))),
        _ => {
            macro_rules! get_primitive_builder {
                ($type:ty) => {{
                    let builder = FixedSizeListPrimitiveChunkedBuilder::<$type>::new(
                        name,
                        list_capacity,
                        inner_type_logical.clone(),
                        inner_size,
                    );
                    Box::new(builder)
                }};
            }
            macro_rules! get_bool_builder {
                () => {{
                    let builder =
                        FixedSizeListBooleanChunkedBuilder::new(&name, list_capacity, inner_size);
                    Box::new(builder)
                }};
            }
            macro_rules! get_utf8_builder {
                () => {{
                    let builder =
                        FixedSizeListUtf8ChunkedBuilder::new(&name, list_capacity, 5 * inner_size);
                    Box::new(builder)
                }};
            }
            macro_rules! get_binary_builder {
                () => {{
                    let builder = FixedSizeListBinaryChunkedBuilder::new(
                        &name,
                        list_capacity,
                        5 * inner_size,
                    );
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

pub struct AnonymousFixedSizeListBuilder<'a> {
    name: String,
    builder: AnonymousBuilder<'a>,
    fast_explode: bool,
    inner_dtype: Option<DataType>,
    inner_size: usize,
}

impl Default for AnonymousFixedSizeListBuilder<'_> {
    fn default() -> Self {
        Self::new("", 0, None, 0)
    }
}

impl<'a> AnonymousFixedSizeListBuilder<'a> {
    pub fn new(
        name: &str,
        capacity: usize,
        inner_dtype: Option<DataType>,
        inner_size: usize,
    ) -> Self {
        Self {
            name: name.into(),
            builder: AnonymousBuilder::new(capacity, inner_size),
            fast_explode: true,
            inner_dtype,
            inner_size,
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

    pub fn finish(&mut self) -> FixedSizeListChunked {
        let inner_size = self.inner_size;
        // don't use self from here on one
        let slf = std::mem::take(self);
        if slf.builder.is_empty() {
            FixedSizeListChunked::full_null_with_dtype(
                &slf.name,
                0,
                &slf.inner_dtype.unwrap_or(DataType::Null),
                inner_size
            )
        } else {
            let inner_dtype_physical = slf
                .inner_dtype
                .as_ref()
                .map(|dt| dt.to_physical().to_arrow());
            let arr = slf.builder.finish(inner_dtype_physical.as_ref()).unwrap();

            let list_dtype_logical = match slf.inner_dtype {
                None => DataType::from(arr.data_type()),
                Some(dt) => DataType::List(Box::new(dt)),
            };
            let mut ca = unsafe { FixedSizeListChunked::from_chunks("", vec![Box::new(arr)]) };

            if slf.fast_explode {
                ca.set_fast_explode();
            }

            ca.field = Arc::new(Field::new(&slf.name, list_dtype_logical));
            ca
        }
    }
}

pub struct AnonymousOwnedFixedSizeListBuilder {
    name: String,
    builder: AnonymousBuilder<'static>,
    owned: Vec<Series>,
    inner_dtype: Option<DataType>,
    fast_explode: bool,
}

impl Default for AnonymousOwnedFixedSizeListBuilder {
    fn default() -> Self {
        Self::new("", 0, None, 0)
    }
}

impl FixedSizeListBuilderTrait for AnonymousOwnedFixedSizeListBuilder {
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

    fn finish(&mut self) -> FixedSizeListChunked {
        // don't use self from here on one
        let slf = std::mem::take(self);
        let inner_dtype_physical = slf
            .inner_dtype
            .as_ref()
            .map(|dt| dt.to_physical().to_arrow());
        let arr = slf.builder.finish(inner_dtype_physical.as_ref()).unwrap();

        let list_dtype_logical = match slf.inner_dtype {
            None => DataType::from(arr.data_type()),
            Some(dt) => DataType::List(Box::new(dt)),
        };
        // safety: same type
        let mut ca = unsafe { FixedSizeListChunked::from_chunks("", vec![Box::new(arr)]) };

        if slf.fast_explode {
            ca.set_fast_explode();
        }

        ca.field = Arc::new(Field::new(&slf.name, list_dtype_logical));
        ca
    }
}

impl AnonymousOwnedFixedSizeListBuilder {
    pub fn new(
        name: &str,
        capacity: usize,
        inner_dtype: Option<DataType>,
        inner_size: usize,
    ) -> Self {
        Self {
            name: name.into(),
            builder: AnonymousBuilder::new(capacity, inner_size),
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
