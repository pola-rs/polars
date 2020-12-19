use crate::chunked_array::builder::get_list_builder;
use crate::prelude::*;

fn chunk_shift_helper<T>(
    ca: &ChunkedArray<T>,
    builder: &mut PrimitiveChunkedBuilder<T>,
    amount: usize,
    skip: usize,
) where
    T: PolarsNumericType,
    T::Native: Copy,
{
    if ca.null_count() == 0 {
        ca.into_no_null_iter()
            .skip(skip)
            .take(amount)
            .for_each(|v| builder.append_value(v))
    } else {
        ca.into_iter()
            .skip(skip)
            .take(amount)
            .for_each(|opt| builder.append_option(opt));
    }
}

impl<T> ChunkShiftFill<T, Option<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Copy,
{
    fn shift_and_fill(
        &self,
        periods: i32,
        fill_value: Option<T::Native>,
    ) -> Result<ChunkedArray<T>> {
        if periods.abs() >= self.len() as i32 {
            return Err(PolarsError::OutOfBounds(
                format!("The value of parameter `periods`: {} in the shift operation is larger than the length of the ChunkedArray: {}", periods, self.len()).into()));
        }
        let mut builder = PrimitiveChunkedBuilder::<T>::new(self.name(), self.len());
        let amount = self.len() - periods.abs() as usize;

        // Fill the front of the array
        if periods > 0 {
            for _ in 0..periods {
                builder.append_option(fill_value)
            }
            chunk_shift_helper(self, &mut builder, amount, 0);
        // Fill the back of the array
        } else {
            chunk_shift_helper(self, &mut builder, amount, periods.abs() as usize);
            for _ in 0..periods.abs() {
                builder.append_option(fill_value)
            }
        }
        Ok(builder.finish())
    }
}
impl<T> ChunkShift<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Copy,
{
    fn shift(&self, periods: i32) -> Result<ChunkedArray<T>> {
        self.shift_and_fill(periods, None)
    }
}

macro_rules! impl_shift {
    // append_method and append_fn do almost the same. Only for list type, the closure
    // accepts an owned value, while fill_value is a reference. That's why we have two options.
    ($self:ident, $builder:ident, $periods:ident, $fill_value:ident,
    $append_method:ident, $append_fn:expr) => {{
        let amount = $self.len() - $periods.abs() as usize;
        let skip = $periods.abs() as usize;

        // Fill the front of the array
        if $periods > 0 {
            for _ in 0..$periods {
                $builder.$append_method($fill_value)
            }
            $self
                .into_iter()
                .take(amount)
                .for_each(|opt| $append_fn(&mut $builder, opt));
        // Fill the back of the array
        } else {
            $self
                .into_iter()
                .skip(skip)
                .take(amount)
                .for_each(|opt| $append_fn(&mut $builder, opt));
            for _ in 0..$periods.abs() {
                $builder.$append_method($fill_value)
            }
        }
        Ok($builder.finish())
    }};
}

impl ChunkShiftFill<BooleanType, Option<bool>> for BooleanChunked {
    fn shift_and_fill(&self, periods: i32, fill_value: Option<bool>) -> Result<BooleanChunked> {
        if periods.abs() >= self.len() as i32 {
            return Err(PolarsError::OutOfBounds(
                format!("The value of parameter `periods`: {} in the shift operation is larger than the length of the ChunkedArray: {}", periods, self.len()).into()));
        }
        let mut builder = PrimitiveChunkedBuilder::<BooleanType>::new(self.name(), self.len());

        fn append_fn(builder: &mut PrimitiveChunkedBuilder<BooleanType>, v: Option<bool>) {
            builder.append_option(v);
        }

        impl_shift!(self, builder, periods, fill_value, append_option, append_fn)
    }
}

impl ChunkShift<BooleanType> for BooleanChunked {
    fn shift(&self, periods: i32) -> Result<Self> {
        self.shift_and_fill(periods, None)
    }
}

impl ChunkShiftFill<Utf8Type, Option<&str>> for Utf8Chunked {
    fn shift_and_fill(&self, periods: i32, fill_value: Option<&str>) -> Result<Utf8Chunked> {
        if periods.abs() >= self.len() as i32 {
            return Err(PolarsError::OutOfBounds(
                format!("The value of parameter `periods`: {} in the shift operation is larger than the length of the ChunkedArray: {}", periods, self.len()).into()));
        }
        let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len());
        fn append_fn(builder: &mut Utf8ChunkedBuilder, v: Option<&str>) {
            builder.append_option(v);
        }

        impl_shift!(self, builder, periods, fill_value, append_option, append_fn)
    }
}

impl ChunkShift<Utf8Type> for Utf8Chunked {
    fn shift(&self, periods: i32) -> Result<Self> {
        self.shift_and_fill(periods, None)
    }
}

impl ChunkShiftFill<ListType, Option<&Series>> for ListChunked {
    fn shift_and_fill(&self, periods: i32, fill_value: Option<&Series>) -> Result<ListChunked> {
        if periods.abs() >= self.len() as i32 {
            return Err(PolarsError::OutOfBounds(
                format!("The value of parameter `periods`: {} in the shift operation is larger than the length of the ChunkedArray: {}", periods, self.len()).into()));
        }
        let dt = self.get_inner_dtype();
        let mut builder = get_list_builder(dt, self.len(), self.name());
        fn append_fn(builder: &mut Box<dyn ListBuilderTrait>, v: Option<&Series>) {
            builder.append_opt_series(v);
        }

        let amount = self.len() - periods.abs() as usize;
        let skip = periods.abs() as usize;

        // Fill the front of the array
        if periods > 0 {
            for _ in 0..periods {
                builder.append_opt_series(fill_value)
            }
            self.into_iter()
                .take(amount)
                .for_each(|opt| append_fn(&mut builder, opt.as_ref()));
            // Fill the back of the array
        } else {
            self.into_iter()
                .skip(skip)
                .take(amount)
                .for_each(|opt| append_fn(&mut builder, opt.as_ref()));
            for _ in 0..periods.abs() {
                builder.append_opt_series(fill_value)
            }
        }
        Ok(builder.finish())
    }
}

impl ChunkShift<ListType> for ListChunked {
    fn shift(&self, periods: i32) -> Result<Self> {
        self.shift_and_fill(periods, None)
    }
}

impl<T> ChunkShiftFill<ObjectType<T>, Option<ObjectType<T>>> for ObjectChunked<T> {
    fn shift_and_fill(
        &self,
        _periods: i32,
        _fill_value: Option<ObjectType<T>>,
    ) -> Result<ChunkedArray<ObjectType<T>>> {
        todo!()
    }
}
impl<T> ChunkShift<ObjectType<T>> for ObjectChunked<T> {
    fn shift(&self, periods: i32) -> Result<Self> {
        self.shift_and_fill(periods, None)
    }
}
