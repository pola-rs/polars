use super::*;

/// A list builder that is told the shape of its values by the first series appended to it, rather
/// than at construction.
///
/// This is what builds the lists whose values are themselves nested — structs, lists and arrays —
/// where there is no typed builder to reach for.
pub struct AnonymousOwnedListBuilder {
    name: PlSmallStr,
    /// The builder, once the shape of the values is known.
    ///
    /// Until a series arrives that says what the values are, there is nothing to build them with:
    /// the rows appended in the meantime are all made of nulls, and are held in `pending` until
    /// there is a builder to replay them into.
    builder: Option<PlListArrayBuilder>,
    /// The rows appended before the builder existed: `None` is a null row, and `Some(n)` a valid
    /// row covering `n` nulls.
    pending: Vec<Option<usize>>,
    capacity: usize,
    inner_dtype: Option<DataType>,
    fast_explode: bool,
}

impl Default for AnonymousOwnedListBuilder {
    fn default() -> Self {
        Self::new(PlSmallStr::EMPTY, 0, None)
    }
}

impl AnonymousOwnedListBuilder {
    pub fn new(name: PlSmallStr, capacity: usize, inner_dtype: Option<DataType>) -> Self {
        Self {
            name,
            builder: None,
            pending: Vec::new(),
            capacity,
            inner_dtype,
            fast_explode: true,
        }
    }

    #[inline]
    pub fn append_empty(&mut self) {
        self.fast_explode = false;
        self.append_nulls(0);
    }

    /// Appends one row covering `length` nulls, which is all a row can be made of before the shape
    /// of the values is known.
    fn append_nulls(&mut self, length: usize) {
        match &mut self.builder {
            Some(builder) => {
                builder.values_mut().extend_nulls(length);
                builder.finish_row();
            },
            None => self.pending.push(Some(length)),
        }
    }

    /// The builder, made to build values shaped like `dtype` if it does not exist yet.
    fn builder_for(&mut self, dtype: &DataType) -> &mut PlListArrayBuilder {
        self.builder.get_or_insert_with(|| {
            // A builder is shaped like the array it builds, which an empty chunk of the physical
            // type is enough to ask for.
            let values = builder_like(&*new_empty_chunk(dtype));
            let mut builder = PlListArrayBuilder::with_capacity(values, self.capacity);

            // The rows that were appended before the shape was known go in first, so that the
            // ones appended after them still follow them.
            for row in self.pending.drain(..) {
                match row {
                    None => builder.extend_nulls(1),
                    Some(length) => {
                        builder.values_mut().extend_nulls(length);
                        builder.finish_row();
                    },
                }
            }
            builder
        })
    }
}

impl ListBuilderTrait for AnonymousOwnedListBuilder {
    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        match (s.dtype(), &self.inner_dtype) {
            (DataType::Null, _) => {},
            (dt, None) => self.inner_dtype = Some(dt.clone()),
            (dt, Some(set_dt)) => {
                polars_ensure!(dt == set_dt, ComputeError: "dtypes don't match, got {}, expected: {}", dt.pretty_format(), set_dt.pretty_format());
            },
        }

        if s.is_empty() {
            self.append_empty();
        } else if s.dtype().is_null() {
            // A series of nulls says nothing about the shape of the values, so its elements are
            // appended as nulls whether or not there is a builder to append them to yet.
            self.append_nulls(s.len());
        } else {
            let dtype = s.dtype().clone();
            let builder = self.builder_for(&dtype);
            // The chunks are appended whole, so each of them stays in whatever representation it
            // is in, and a series of several chunks does not have to be rechunked first.
            let values = builder.values_mut();
            for chunk in s.chunks() {
                values.extend(&**chunk, ShareStrategy::Always);
            }
            builder.finish_row();
        }
        Ok(())
    }

    #[inline]
    fn append_null(&mut self) {
        self.fast_explode = false;
        match &mut self.builder {
            Some(builder) => builder.extend_nulls(1),
            None => self.pending.push(None),
        }
    }

    fn finish(&mut self) -> ListChunked {
        // Nothing said what the values are, so they are nulls: either that is what was appended,
        // or nothing was.
        let inner_dtype = std::mem::take(&mut self.inner_dtype).unwrap_or(DataType::Null);
        let arr = self.builder_for(&inner_dtype).freeze_reset();

        let mut ca = ListChunked::with_chunk(PlSmallStr::EMPTY, arr);
        if self.fast_explode {
            ca.set_fast_explode();
        }
        ca.field = Arc::new(Field::new(
            self.name.clone(),
            DataType::List(Box::new(inner_dtype)),
        ));
        ca
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elements(ca: &ListChunked) -> Vec<Option<Vec<Option<i32>>>> {
        ca.amortized_iter()
            .map(|row| row.map(|row| row.as_ref().i32().unwrap().iter().collect()))
            .collect()
    }

    #[test]
    fn appending_series_nulls_and_empties() {
        let mut builder = AnonymousOwnedListBuilder::new("a".into(), 8, None);
        builder
            .append_series(&Series::new("".into(), [1i32, 2]))
            .unwrap();
        builder.append_null();
        builder.append_empty();
        builder
            .append_series(&Series::new("".into(), [Some(3i32), None]))
            .unwrap();

        let ca = builder.finish();
        assert_eq!(ca.name(), "a");
        assert_eq!(ca.dtype(), &DataType::List(Box::new(DataType::Int32)));
        assert_eq!(
            elements(&ca),
            [
                Some(vec![Some(1), Some(2)]),
                None,
                Some(vec![]),
                Some(vec![Some(3), None]),
            ],
        );
        // A null row and an empty row both leave a hole in the values, so exploding cannot be a
        // matter of dropping the offsets.
        assert!(!ca.get_fast_explode_list());
    }

    /// The shape of the values is not known until a series arrives that says what they are, and
    /// everything appended before then still has to come first.
    #[test]
    fn rows_appended_before_the_dtype_is_known_keep_their_place() {
        let mut builder = AnonymousOwnedListBuilder::new("a".into(), 8, None);
        builder.append_null();
        builder
            .append_series(&Series::new_null("".into(), 2))
            .unwrap();
        builder.append_empty();
        builder
            .append_series(&Series::new("".into(), [7i32]))
            .unwrap();
        // A null series after the fact says nothing about the values either.
        builder
            .append_series(&Series::new_null("".into(), 1))
            .unwrap();

        let ca = builder.finish();
        assert_eq!(ca.dtype(), &DataType::List(Box::new(DataType::Int32)));
        assert_eq!(
            elements(&ca),
            [
                None,
                Some(vec![None, None]),
                Some(vec![]),
                Some(vec![Some(7)]),
                Some(vec![None]),
            ],
        );
    }

    /// Nothing ever said what the values are, so they are nulls.
    #[test]
    fn a_builder_that_is_only_ever_told_nulls_builds_a_list_of_nulls() {
        let mut builder = AnonymousOwnedListBuilder::new("a".into(), 8, None);
        builder.append_null();
        builder
            .append_series(&Series::new_null("".into(), 2))
            .unwrap();

        let ca = builder.finish();
        assert_eq!(ca.dtype(), &DataType::List(Box::new(DataType::Null)));
        assert_eq!(ca.len(), 2);

        let empty = AnonymousOwnedListBuilder::new("a".into(), 0, None).finish();
        assert_eq!(empty.dtype(), &DataType::List(Box::new(DataType::Null)));
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn a_series_of_the_wrong_dtype_is_rejected() {
        let mut builder = AnonymousOwnedListBuilder::new("a".into(), 8, None);
        builder
            .append_series(&Series::new("".into(), [1i32]))
            .unwrap();
        assert!(
            builder
                .append_series(&Series::new("".into(), ["x"]))
                .is_err()
        );
    }

    /// A series of several chunks is appended chunk by chunk rather than rechunked first.
    #[test]
    fn a_multi_chunk_series_is_one_row() {
        let mut s = Series::new("".into(), [1i32, 2]);
        s.append(&Series::new("".into(), [3i32])).unwrap();
        assert_eq!(s.n_chunks(), 2);

        let mut builder = AnonymousOwnedListBuilder::new("a".into(), 8, None);
        builder.append_series(&s).unwrap();

        let ca = builder.finish();
        assert_eq!(elements(&ca), [Some(vec![Some(1), Some(2), Some(3)])]);
    }
}
