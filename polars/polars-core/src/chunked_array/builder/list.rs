use super::*;
use polars_arrow::prelude::PolarsArray;

pub trait ListBuilderTrait {
    fn append_opt_series(&mut self, opt_s: Option<&Series>);
    fn append_series(&mut self, s: &Series);
    fn append_null(&mut self);
    fn finish(&mut self) -> ListChunked;
}

pub struct ListPrimitiveChunkedBuilder<T>
where
    T: NumericNative,
{
    pub builder: LargePrimitiveBuilder<T>,
    field: Field,
    fast_explode: bool,
}

macro_rules! finish_list_builder {
    ($self:ident) => {{
        let arr = $self.builder.as_arc();
        let mut ca = ListChunked {
            field: Arc::new($self.field.clone()),
            chunks: vec![arr],
            phantom: PhantomData,
            categorical_map: None,
            ..Default::default()
        };
        if $self.fast_explode {
            ca.set_fast_explode()
        }
        ca
    }};
}

impl<T> ListPrimitiveChunkedBuilder<T>
where
    T: NumericNative,
{
    pub fn new(
        name: &str,
        capacity: usize,
        values_capacity: usize,
        logical_type: DataType,
    ) -> Self {
        let values = MutablePrimitiveArray::<T>::with_capacity(values_capacity);
        let builder = LargePrimitiveBuilder::<T>::new_with_capacity(values, capacity);
        let field = Field::new(name, DataType::List(Box::new(logical_type)));

        Self {
            builder,
            field,
            fast_explode: true,
        }
    }

    pub fn append_slice(&mut self, opt_v: Option<&[T]>) {
        match opt_v {
            Some(items) => {
                let values = self.builder.mut_values();
                values.extend_from_slice(items);
                self.builder.try_push_valid().unwrap();

                if items.is_empty() {
                    self.fast_explode = false;
                }
            }
            None => {
                self.builder.push_null();
            }
        }
    }
    /// Appends from an iterator over values
    #[inline]
    pub fn append_iter_values<I: Iterator<Item = T> + TrustedLen>(&mut self, iter: I) {
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
    pub fn append_iter<I: Iterator<Item = Option<T>> + TrustedLen>(&mut self, iter: I) {
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
    T: NumericNative,
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
        let arrays = s.chunks();
        let values = self.builder.mut_values();

        arrays.iter().for_each(|x| {
            let arr = x.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();

            if !arr.has_validity() {
                values.extend_from_slice(arr.values().as_slice())
            } else {
                // Safety:
                // Arrow arrays are trusted length iterators.
                unsafe { values.extend_trusted_len_unchecked(arr.into_iter()) }
            }
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

    #[inline]
    pub fn append_iter<'a, I: Iterator<Item = Option<&'a str>> + TrustedLen>(&mut self, iter: I) {
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

    #[inline]
    fn append_series(&mut self, s: &Series) {
        if s.is_empty() {
            self.fast_explode = false;
        }
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
        if ca.is_empty() {
            self.fast_explode = false;
        }
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
    let physical_type = dt.to_physical();

    macro_rules! get_primitive_builder {
        ($type:ty) => {{
            let builder = ListPrimitiveChunkedBuilder::<$type>::new(
                &name,
                list_capacity,
                value_capacity,
                dt.clone(),
            );
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
    match_dtype_to_physical_apply_macro!(
        physical_type,
        get_primitive_builder,
        get_utf8_builder,
        get_bool_builder
    )
}
