pub mod categorical;
pub use self::categorical::CategoricalChunkedBuilder;
use crate::{
    prelude::*,
    utils::{get_iter_capacity, NoNull},
};
use arrow::{array::*, bitmap::Bitmap};
use std::borrow::Cow;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::sync::Arc;

// N: the value type; T: the sentinel type
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
    array_builder: MutableBooleanArray,
    field: Field,
}

impl ChunkedBuilder<bool, BooleanType> for BooleanChunkedBuilder {
    /// Appends a value of type `T` into the builder
    #[inline]
    fn append_value(&mut self, v: bool) {
        self.array_builder.push(Some(v));
    }

    /// Appends a null slot into the builder
    #[inline]
    fn append_null(&mut self) {
        self.array_builder.push(None);
    }

    fn finish(self) -> BooleanChunked {
        let arr: BooleanArray = self.array_builder.into();
        let arr = Arc::new(arr) as ArrayRef;

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
            array_builder: MutableBooleanArray::with_capacity(capacity),
            field: Field::new(name, DataType::Boolean),
        }
    }
}

pub struct PrimitiveChunkedBuilder<T>
where
    T: PolarsPrimitiveType,
    T::Native: Default,
{
    array_builder: MutablePrimitiveArray<T::Native>,
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
        self.array_builder.push(Some(v))
    }

    /// Appends a null slot into the builder
    #[inline]
    fn append_null(&mut self) {
        self.array_builder.push(None)
    }

    fn finish(self) -> ChunkedArray<T> {
        ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![self.array_builder.into_arc()],
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
        let array_builder = MutablePrimitiveArray::<T::Native>::with_capacity(capacity)
            .to(T::get_dtype().to_arrow());

        PrimitiveChunkedBuilder {
            array_builder,
            field: Field::new(name, T::get_dtype()),
        }
    }
}

pub struct Utf8ChunkedBuilder {
    pub builder: MutableUtf8Array<i64>,
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
            builder: MutableUtf8Array::<i64>::with_capacities(capacity, bytes_capacity),
            capacity,
            field: Field::new(name, DataType::Utf8),
        }
    }

    /// Appends a value of type `T` into the builder
    #[inline]
    pub fn append_value<S: AsRef<str>>(&mut self, v: S) {
        self.builder.push(Some(v.as_ref()));
    }

    /// Appends a null slot into the builder
    #[inline]
    pub fn append_null(&mut self) {
        self.builder.push::<&str>(None);
    }

    #[inline]
    pub fn append_option<S: AsRef<str>>(&mut self, opt: Option<S>) {
        self.builder.push(opt);
    }

    pub fn finish(self) -> Utf8Chunked {
        let arr = self.builder.into_arc();
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

// Used in polars/src/chunked_array/apply.rs:24 to collect from aligned vecs and null bitmaps
impl<T> FromIterator<(AlignedVec<T::Native>, Option<Bitmap>)> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_iter<I: IntoIterator<Item = (AlignedVec<T::Native>, Option<Bitmap>)>>(iter: I) -> Self {
        let mut chunks = vec![];

        for (values, opt_buffer) in iter {
            chunks.push(to_array::<T>(values, opt_buffer))
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
        let arr = PrimitiveArray::<T::Native>::from_slice(v).to(T::get_dtype().to_arrow());
        ChunkedArray::new_from_chunks(name, vec![Arc::new(arr)])
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

        let mut builder = MutableUtf8Array::<i64>::with_capacities(v.len(), values_size);
        v.iter().for_each(|val| {
            builder.push(Some(val.as_ref()));
        });

        let field = Arc::new(Field::new(name, DataType::Utf8));

        ChunkedArray {
            field,
            chunks: vec![builder.into_arc()],
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
        let mut builder = Utf8ChunkedBuilder::new(name, opt_v.len(), values_size);

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
    T: PolarsNumericType,
{
    pub builder: LargePrimitiveBuilder<T::Native>,
    field: Field,
}

macro_rules! finish_list_builder {
    ($self:ident) => {{
        let arr = $self.builder.as_arc();
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
    T: PolarsNumericType,
{
    pub fn new(name: &str, capacity: usize, values_capacity: usize) -> Self {
        let values = MutablePrimitiveArray::<T::Native>::with_capacity(values_capacity);
        let builder = LargePrimitiveBuilder::<T::Native>::new_with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(T::get_dtype().to_arrow()));

        Self { builder, field }
    }

    pub fn append_slice(&mut self, opt_v: Option<&[T::Native]>) {
        match opt_v {
            Some(items) => {
                let values = self.builder.mut_values();
                // Safety:
                // A slice is a trusted length iterator
                unsafe { values.extend_trusted_len_unchecked(items.iter().map(Some)) }
                self.builder.try_push_valid().unwrap();
            }
            None => {
                self.builder.push_null();
            }
        }
    }

    pub fn append_null(&mut self) {
        self.builder.push_null();
    }
}

impl<T> ListBuilderTrait for ListPrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    #[inline]
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => {
                self.builder.push_null();
            }
        }
    }

    #[inline]
    fn append_null(&mut self) {
        self.builder.push_null();
    }

    #[inline]
    fn append_series(&mut self, s: &Series) {
        let arrays = s.chunks();
        let values = self.builder.mut_values();

        arrays.iter().for_each(|x| {
            let arr = x
                .as_any()
                .downcast_ref::<PrimitiveArray<T::Native>>()
                .unwrap();
            // Safety:
            // Arrow arrays are trusted length iterators.
            unsafe { values.extend_trusted_len_unchecked(arr.into_iter()) }
        });
        self.builder.try_push_valid().unwrap();
    }

    fn finish(&mut self) -> ListChunked {
        finish_list_builder!(self)
    }
}

type LargePrimitiveBuilder<T> = MutableListArray<i64, MutablePrimitiveArray<T>>;
type LargeListUtf8Builder = MutableListArray<i64, MutableUtf8Array<i64>>;
type LargeListBooleanBuilder = MutableListArray<i64, MutableBooleanArray>;

pub struct ListUtf8ChunkedBuilder {
    builder: LargeListUtf8Builder,
    field: Field,
}

impl ListUtf8ChunkedBuilder {
    pub fn new(name: &str, capacity: usize, values_capacity: usize) -> Self {
        let values = MutableUtf8Array::<i64>::with_capacity(values_capacity);
        let builder = LargeListUtf8Builder::new_with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(ArrowDataType::LargeUtf8));

        ListUtf8ChunkedBuilder { builder, field }
    }
}

impl ListBuilderTrait for ListUtf8ChunkedBuilder {
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => {
                self.builder.push_null();
            }
        }
    }

    #[inline]
    fn append_null(&mut self) {
        self.builder.push_null();
    }

    #[inline]
    fn append_series(&mut self, s: &Series) {
        let ca = s.utf8().unwrap();
        let value_builder = self.builder.mut_values();
        value_builder.try_extend(ca).unwrap();
        self.builder.try_push_valid().unwrap();
    }

    fn finish(&mut self) -> ListChunked {
        finish_list_builder!(self)
    }
}

pub struct ListBooleanChunkedBuilder {
    builder: LargeListBooleanBuilder,
    field: Field,
}

impl ListBooleanChunkedBuilder {
    pub fn new(name: &str, capacity: usize, values_capacity: usize) -> Self {
        let values = MutableBooleanArray::with_capacity(values_capacity);
        let builder = LargeListBooleanBuilder::new_with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(ArrowDataType::Boolean));

        Self { builder, field }
    }
}

impl ListBuilderTrait for ListBooleanChunkedBuilder {
    fn append_opt_series(&mut self, opt_s: Option<&Series>) {
        match opt_s {
            Some(s) => self.append_series(s),
            None => {
                self.builder.push_null();
            }
        }
    }

    #[inline]
    fn append_null(&mut self) {
        self.builder.push_null();
    }

    #[inline]
    fn append_series(&mut self, s: &Series) {
        let ca = s.bool().unwrap();
        let value_builder = self.builder.mut_values();
        value_builder.extend(ca);
        self.builder.try_push_valid().unwrap();
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
            let builder =
                ListPrimitiveChunkedBuilder::<$type>::new(&name, list_capacity, value_capacity);
            Box::new(builder)
        }};
    }
    macro_rules! get_bool_builder {
        () => {{
            let builder = ListBooleanChunkedBuilder::new(&name, list_capacity, value_capacity);
            Box::new(builder)
        }};
    }
    macro_rules! get_utf8_builder {
        () => {{
            let builder = ListUtf8ChunkedBuilder::new(&name, list_capacity, 5 * value_capacity);
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
        let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new("a", 10, 5);

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
        let mut builder = ListUtf8ChunkedBuilder::new("a", 10, 10);
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
