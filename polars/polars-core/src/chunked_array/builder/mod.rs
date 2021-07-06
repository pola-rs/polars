pub mod categorical;
pub use self::categorical::CategoricalChunkedBuilder;
use crate::{
    prelude::*,
    utils::{get_iter_capacity, NoNull},
};
pub use arrow::alloc;
use arrow::array::{ArrayData, PrimitiveArray, PrimitiveBuilder};
use arrow::array::{ArrayRef, LargeListBuilder};
use arrow::{array::Array, buffer::Buffer};
use num::Num;
use polars_arrow::prelude::*;
use std::borrow::Cow;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::sync::Arc;

pub trait ChunkedBuilder<N, T> {
    fn append_value(&mut self, val: N);
    fn append_null(&mut self);
    fn append_option(&mut self, opt_val: Option<N>) {
        match opt_val {
            Some(v) => self.append_value(v),
            None => self.append_null(),
        }
    }
    fn finish(self) -> ChunkedArray<T>;
}

pub struct BooleanChunkedBuilder {
    array_builder: BooleanArrayBuilder,
    field: Field,
}

impl ChunkedBuilder<bool, BooleanType> for BooleanChunkedBuilder {
    /// Appends a value of type `T` into the builder
    #[inline]
    fn append_value(&mut self, v: bool) {
        self.array_builder.append_value(v);
    }

    /// Appends a null slot into the builder
    #[inline]
    fn append_null(&mut self) {
        self.array_builder.append_null();
    }

    fn finish(mut self) -> BooleanChunked {
        let arr = Arc::new(self.array_builder.finish());

        ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            phantom: PhantomData,
            categorical_map: None,
            ..Default::default()
        }
    }
}

impl BooleanChunkedBuilder {
    pub fn new(name: &str, capacity: usize) -> Self {
        BooleanChunkedBuilder {
            array_builder: BooleanArrayBuilder::new(capacity),
            field: Field::new(name, DataType::Boolean),
        }
    }
}

pub struct PrimitiveChunkedBuilder<T>
where
    T: PolarsPrimitiveType,
    T::Native: Default,
{
    array_builder: PrimitiveBuilder<T>,
    field: Field,
}

impl<T> ChunkedBuilder<T::Native, T> for PrimitiveChunkedBuilder<T>
where
    T: PolarsPrimitiveType,
    T::Native: Default,
{
    /// Appends a value of type `T` into the builder
    #[inline]
    fn append_value(&mut self, v: T::Native) {
        self.array_builder.append_value(v).unwrap()
    }

    /// Appends a null slot into the builder
    #[inline]
    fn append_null(&mut self) {
        self.array_builder.append_null().unwrap()
    }

    fn finish(mut self) -> ChunkedArray<T> {
        let arr = Arc::new(self.array_builder.finish());

        ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            phantom: PhantomData,
            categorical_map: None,
            ..Default::default()
        }
    }
}

impl<T> PrimitiveChunkedBuilder<T>
where
    T: PolarsPrimitiveType,
{
    pub fn new(name: &str, capacity: usize) -> Self {
        PrimitiveChunkedBuilder {
            array_builder: PrimitiveBuilder::<T>::new(capacity),
            field: Field::new(name, T::get_dtype()),
        }
    }
}

pub struct Utf8ChunkedBuilder {
    pub builder: LargeStringBuilder,
    pub capacity: usize,
    field: Field,
}

impl Utf8ChunkedBuilder {
    /// Create a new UtfChunkedBuilder
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of string elements in the final array.
    /// * `bytes_capacity` - Number of bytes needed to store the string values.
    pub fn new(name: &str, capacity: usize, bytes_capacity: usize) -> Self {
        Utf8ChunkedBuilder {
            builder: LargeStringBuilder::with_capacity(bytes_capacity, capacity),
            capacity,
            field: Field::new(name, DataType::Utf8),
        }
    }

    /// Appends a value of type `T` into the builder
    #[inline]
    pub fn append_value<S: AsRef<str>>(&mut self, v: S) {
        self.builder.append_value(v.as_ref()).unwrap();
    }

    /// Appends a null slot into the builder
    #[inline]
    pub fn append_null(&mut self) {
        self.builder.append_null().unwrap();
    }

    #[inline]
    pub fn append_option<S: AsRef<str>>(&mut self, opt: Option<S>) {
        match opt {
            Some(s) => self.append_value(s.as_ref()),
            None => self.append_null(),
        }
    }

    pub fn finish(mut self) -> Utf8Chunked {
        let arr = Arc::new(self.builder.finish());
        ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            phantom: PhantomData,
            categorical_map: None,
            ..Default::default()
        }
    }
}

pub struct Utf8ChunkedBuilderCow {
    builder: Utf8ChunkedBuilder,
}

impl Utf8ChunkedBuilderCow {
    pub fn new(name: &str, capacity: usize) -> Self {
        Utf8ChunkedBuilderCow {
            builder: Utf8ChunkedBuilder::new(name, capacity, capacity),
        }
    }
}

impl ChunkedBuilder<Cow<'_, str>, Utf8Type> for Utf8ChunkedBuilderCow {
    #[inline]
    fn append_value(&mut self, val: Cow<'_, str>) {
        self.builder.append_value(val.as_ref())
    }

    #[inline]
    fn append_null(&mut self) {
        self.builder.append_null()
    }

    fn finish(self) -> ChunkedArray<Utf8Type> {
        self.builder.finish()
    }
}

/// Get the null count and the null bitmap of the arrow array
pub fn get_bitmap<T: Array + ?Sized>(arr: &T) -> (usize, Option<&Buffer>) {
    let data = arr.data();
    (data.null_count(), data.null_buffer())
}

// Used in polars/src/chunked_array/apply.rs:24 to collect from aligned vecs and null bitmaps
impl<T> FromIterator<(AlignedVec<T::Native>, Option<Buffer>)> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_iter<I: IntoIterator<Item = (AlignedVec<T::Native>, Option<Buffer>)>>(iter: I) -> Self {
        let mut chunks = vec![];

        for (values, opt_buffer) in iter {
            let arr = values.into_primitive_array::<T>(opt_buffer);
            chunks.push(Arc::new(arr) as ArrayRef)
        }
        ChunkedArray::new_from_chunks("from_iter", chunks)
    }
}

pub trait NewChunkedArray<T, N> {
    fn new_from_slice(name: &str, v: &[N]) -> Self;
    fn new_from_opt_slice(name: &str, opt_v: &[Option<N>]) -> Self;

    /// Create a new ChunkedArray from an iterator.
    fn new_from_opt_iter(name: &str, it: impl Iterator<Item = Option<N>>) -> Self;

    /// Create a new ChunkedArray from an iterator.
    fn new_from_iter(name: &str, it: impl Iterator<Item = N>) -> Self;
}

impl<T> NewChunkedArray<T, T::Native> for ChunkedArray<T>
where
    T: PolarsPrimitiveType,
{
    fn new_from_slice(name: &str, v: &[T::Native]) -> Self {
        let array_data = ArrayData::builder(T::DATA_TYPE)
            .len(v.len())
            .add_buffer(Buffer::from_slice_ref(&v))
            .build();
        ChunkedArray::new_from_chunks(name, vec![Arc::new(PrimitiveArray::<T>::from(array_data))])
    }

    fn new_from_opt_slice(name: &str, opt_v: &[Option<T::Native>]) -> Self {
        Self::new_from_opt_iter(name, opt_v.iter().copied())
    }

    fn new_from_opt_iter(
        name: &str,
        it: impl Iterator<Item = Option<T::Native>>,
    ) -> ChunkedArray<T> {
        let mut builder = PrimitiveChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    /// Create a new ChunkedArray from an iterator.
    fn new_from_iter(name: &str, it: impl Iterator<Item = T::Native>) -> ChunkedArray<T> {
        let ca: NoNull<ChunkedArray<_>> = it.collect();
        let mut ca = ca.into_inner();
        ca.rename(name);
        ca
    }
}

impl NewChunkedArray<BooleanType, bool> for BooleanChunked {
    fn new_from_slice(name: &str, v: &[bool]) -> Self {
        Self::new_from_iter(name, v.iter().copied())
    }

    fn new_from_opt_slice(name: &str, opt_v: &[Option<bool>]) -> Self {
        Self::new_from_opt_iter(name, opt_v.iter().copied())
    }

    fn new_from_opt_iter(
        name: &str,
        it: impl Iterator<Item = Option<bool>>,
    ) -> ChunkedArray<BooleanType> {
        let mut builder = BooleanChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    /// Create a new ChunkedArray from an iterator.
    fn new_from_iter(name: &str, it: impl Iterator<Item = bool>) -> ChunkedArray<BooleanType> {
        let mut ca: ChunkedArray<_> = it.collect();
        ca.rename(name);
        ca
    }
}

impl<S> NewChunkedArray<Utf8Type, S> for Utf8Chunked
where
    S: AsRef<str>,
{
    fn new_from_slice(name: &str, v: &[S]) -> Self {
        let values_size = v.iter().fold(0, |acc, s| acc + s.as_ref().len());

        let mut builder = LargeStringBuilder::with_capacity(values_size, v.len());
        v.iter().for_each(|val| {
            builder.append_value(val.as_ref()).unwrap();
        });

        let field = Arc::new(Field::new(name, DataType::Utf8));

        ChunkedArray {
            field,
            chunks: vec![Arc::new(builder.finish())],
            phantom: PhantomData,
            categorical_map: None,
            ..Default::default()
        }
    }

    fn new_from_opt_slice(name: &str, opt_v: &[Option<S>]) -> Self {
        let values_size = opt_v.iter().fold(0, |acc, s| match s {
            Some(s) => acc + s.as_ref().len(),
            None => acc,
        });
        let mut builder = Utf8ChunkedBuilder::new(name, values_size, opt_v.len());

        opt_v.iter().for_each(|opt| match opt {
            Some(v) => builder.append_value(v.as_ref()),
            None => builder.append_null(),
        });
        builder.finish()
    }

    fn new_from_opt_iter(name: &str, it: impl Iterator<Item = Option<S>>) -> Self {
        let cap = get_iter_capacity(&it);
        let mut builder = Utf8ChunkedBuilder::new(name, cap, cap * 5);
        it.for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    /// Create a new ChunkedArray from an iterator.
    fn new_from_iter(name: &str, it: impl Iterator<Item = S>) -> Self {
        let cap = get_iter_capacity(&it);
        let mut builder = Utf8ChunkedBuilder::new(name, cap, cap * 5);
        it.for_each(|v| builder.append_value(v));
        builder.finish()
    }
}

pub trait ListBuilderTrait {
    fn append_opt_series(&mut self, opt_s: Option<&Series>);
    fn append_series(&mut self, s: &Series);
    fn append_null(&mut self);
    fn finish(&mut self) -> ListChunked;
}

pub struct ListPrimitiveChunkedBuilder<T>
where
    T: PolarsPrimitiveType,
{
    pub builder: LargeListBuilder<PrimitiveBuilder<T>>,
    field: Field,
}

macro_rules! finish_list_builder {
    ($self:ident) => {{
        let arr = Arc::new($self.builder.finish());
        ListChunked {
            field: Arc::new($self.field.clone()),
            chunks: vec![arr],
            phantom: PhantomData,
            categorical_map: None,
            ..Default::default()
        }
    }};
}

impl<T> ListPrimitiveChunkedBuilder<T>
where
    T: PolarsPrimitiveType,
{
    pub fn new(name: &str, values_builder: PrimitiveBuilder<T>, capacity: usize) -> Self {
        let builder = LargeListBuilder::with_capacity(values_builder, capacity);
        let field = Field::new(name, DataType::List(T::get_dtype().to_arrow()));

        ListPrimitiveChunkedBuilder { builder, field }
    }

    pub fn append_slice(&mut self, opt_v: Option<&[T::Native]>) {
        match opt_v {
            Some(v) => {
                self.builder.values().append_slice(v).unwrap();
                self.builder.append(true).expect("should not fail");
            }
            None => {
                self.builder.append(false).expect("should not fail");
            }
        }
    }

    pub fn append_null(&mut self) {
        self.builder.append(false).expect("should not fail");
    }
}

impl<T> ListBuilderTrait for ListPrimitiveChunkedBuilder<T>
where
    T: PolarsPrimitiveType,
    T::Native: Num,
{
    #[inline]
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => {
                self.builder.append(false).unwrap();
            }
        }
    }

    #[inline]
    fn append_null(&mut self) {
        let builder = self.builder.values();
        builder.append_null().unwrap();
        self.builder.append(true).unwrap();
    }

    #[inline]
    fn append_series(&mut self, s: &Series) {
        let builder = self.builder.values();
        let arrays = s.chunks();
        if s.null_count() == s.len() {
            for _ in 0..s.len() {
                builder.append_null().unwrap();
            }
        } else if s.is_empty() {
            return;
        } else {
            for a in arrays {
                let values = a.get_values::<T>();
                // we would like to check if array has no null values.
                // however at the time of writing there is a bug in append_slice, because it does not update
                // the null bitmap
                if s.null_count() == 0 {
                    builder.append_slice(values).unwrap();
                } else {
                    values.iter().enumerate().for_each(|(idx, v)| {
                        if a.is_valid(idx) {
                            builder.append_value(*v).unwrap();
                        } else {
                            builder.append_null().unwrap();
                        }
                    });
                }
            }
        }
        self.builder.append(true).unwrap();
    }

    fn finish(&mut self) -> ListChunked {
        finish_list_builder!(self)
    }
}

pub struct ListUtf8ChunkedBuilder {
    builder: LargeListBuilder<LargeStringBuilder>,
    field: Field,
}

impl ListUtf8ChunkedBuilder {
    pub fn new(name: &str, values_builder: LargeStringBuilder, capacity: usize) -> Self {
        let builder = LargeListBuilder::with_capacity(values_builder, capacity);
        let field = Field::new(name, DataType::List(ArrowDataType::LargeUtf8));

        ListUtf8ChunkedBuilder { builder, field }
    }
}

impl ListBuilderTrait for ListUtf8ChunkedBuilder {
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => {
                self.builder.append(false).unwrap();
            }
        }
    }

    #[inline]
    fn append_null(&mut self) {
        let builder = self.builder.values();
        builder.append_null().unwrap();
        self.builder.append(true).unwrap();
    }

    #[inline]
    fn append_series(&mut self, s: &Series) {
        let ca = s.utf8().unwrap();
        let value_builder = self.builder.values();
        for s in ca {
            match s {
                Some(s) => value_builder.append_value(s).unwrap(),
                None => value_builder.append_null().unwrap(),
            };
        }
        self.builder.append(true).unwrap();
    }

    fn finish(&mut self) -> ListChunked {
        finish_list_builder!(self)
    }
}

pub struct ListBooleanChunkedBuilder {
    builder: LargeListBuilder<BooleanArrayBuilder>,
    field: Field,
}

impl ListBooleanChunkedBuilder {
    pub fn new(name: &str, values_builder: BooleanArrayBuilder, capacity: usize) -> Self {
        let builder = LargeListBuilder::with_capacity(values_builder, capacity);
        let field = Field::new(name, DataType::List(ArrowDataType::Boolean));

        Self { builder, field }
    }
}

impl ListBuilderTrait for ListBooleanChunkedBuilder {
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => {
                self.builder.append(false).unwrap();
            }
        }
    }

    #[inline]
    fn append_null(&mut self) {
        let builder = self.builder.values();
        builder.append_null();
        self.builder.append(true).unwrap();
    }

    #[inline]
    fn append_series(&mut self, s: &Series) {
        let ca = s.bool().unwrap();
        let value_builder = self.builder.values();
        for s in ca {
            match s {
                Some(s) => value_builder.append_value(s),
                None => value_builder.append_null(),
            };
        }
        self.builder.append(true).unwrap();
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
) -> Box<dyn ListBuilderTrait> {
    macro_rules! get_primitive_builder {
        ($type:ty) => {{
            let values_builder = PrimitiveBuilder::<$type>::new(value_capacity);
            let builder = ListPrimitiveChunkedBuilder::new(&name, values_builder, list_capacity);
            Box::new(builder)
        }};
    }
    macro_rules! get_bool_builder {
        () => {{
            let values_builder = BooleanArrayBuilder::new(value_capacity);
            let builder = ListBooleanChunkedBuilder::new(&name, values_builder, list_capacity);
            Box::new(builder)
        }};
    }
    macro_rules! get_utf8_builder {
        () => {{
            let values_builder =
                LargeStringBuilder::with_capacity(value_capacity * 5, value_capacity);
            let builder = ListUtf8ChunkedBuilder::new(&name, values_builder, list_capacity);
            Box::new(builder)
        }};
    }
    match_arrow_data_type_apply_macro!(
        dt,
        get_primitive_builder,
        get_utf8_builder,
        get_bool_builder
    )
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{reset_string_cache, toggle_string_cache};

    #[test]
    fn test_primitive_builder() {
        let mut builder = PrimitiveChunkedBuilder::<UInt32Type>::new("foo", 6);
        let values = &[Some(1), None, Some(2), Some(3), None, Some(4)];
        for val in values {
            builder.append_option(*val);
        }
        let ca = builder.finish();
        assert_eq!(Vec::from(&ca), values);
    }

    #[test]
    fn test_list_builder() {
        let values_builder = PrimitiveBuilder::<Int32Type>::new(10);
        let mut builder = ListPrimitiveChunkedBuilder::new("a", values_builder, 10);

        // create a series containing two chunks
        let mut s1 = Int32Chunked::new_from_slice("a", &[1, 2, 3]).into_series();
        let s2 = Int32Chunked::new_from_slice("b", &[4, 5, 6]).into_series();
        s1.append(&s2).unwrap();

        builder.append_series(&s1);
        builder.append_series(&s2);
        let ls = builder.finish();
        if let AnyValue::List(s) = ls.get_any_value(0) {
            // many chunks are aggregated to one in the ListArray
            assert_eq!(s.len(), 6)
        } else {
            panic!()
        }
        if let AnyValue::List(s) = ls.get_any_value(1) {
            assert_eq!(s.len(), 3)
        } else {
            panic!()
        }
        // test list collect
        let out = [&s1, &s2].iter().copied().collect::<ListChunked>();
        assert_eq!(out.get(0).unwrap().len(), 6);
        assert_eq!(out.get(1).unwrap().len(), 3);
    }

    #[test]
    fn test_list_str_builder() {
        let mut builder =
            ListUtf8ChunkedBuilder::new("a", LargeStringBuilder::with_capacity(10, 10), 10);
        builder.append_series(&Series::new("", &["foo", "bar"]));
        let ca = builder.finish();
        dbg!(ca);
    }

    #[test]
    fn test_categorical_builder() {
        let _lock = crate::SINGLE_LOCK.lock();
        for b in &[false, true] {
            reset_string_cache();
            toggle_string_cache(*b);

            // Use 2 builders to check if the global string cache
            // does not interfere with the index mapping
            let mut builder1 = CategoricalChunkedBuilder::new("foo", 10);
            let mut builder2 = CategoricalChunkedBuilder::new("foo", 10);
            builder1.from_iter(vec![None, Some("hello"), Some("vietnam")]);
            builder2.from_iter(vec![Some("hello"), None, Some("world")].into_iter());

            let ca = builder1.finish();
            let v = AnyValue::Null;
            assert_eq!(ca.get_any_value(0), v);
            let v = AnyValue::Utf8("hello");
            assert_eq!(ca.get_any_value(1), v);
            let v = AnyValue::Utf8("vietnam");
            assert_eq!(ca.get_any_value(2), v);

            let ca = builder2.finish();
            let v = AnyValue::Utf8("hello");
            assert_eq!(ca.get_any_value(0), v);
            let v = AnyValue::Null;
            assert_eq!(ca.get_any_value(1), v);
            let v = AnyValue::Utf8("world");
            assert_eq!(ca.get_any_value(2), v);
        }
    }
}
