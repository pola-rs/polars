use super::*;

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn primitive_compare_scalar<Rhs: ToPrimitive>(
        &self,
        rhs: Rhs,
        f: impl Fn(&PrimitiveArray<T::Native>, &dyn Scalar) -> BooleanArray,
    ) -> BooleanChunked {
        let rhs: T::Native = NumCast::from(rhs).unwrap();
        let scalar = PrimitiveScalar::new(T::get_dtype().to_arrow(), Some(rhs));
        self.apply_kernel_cast(&|arr| Box::new(f(arr, &scalar)))
    }
}

/// Splits the ChunkedArray into a lower part, where is_lower returns true, and
/// an upper part where it returns false, and returns a mask where the lower part
/// has value lower_part, and the upper part !lower_part.
/// The ChunkedArray is assumed to be sorted w.r.t. is_lower, that is, is_lower
/// first always returns true, and then always returns false.
fn partition_mask<T: PolarsNumericType, F>(
    ca: &ChunkedArray<T>,
    lower_part: bool,
    is_lower: F,
) -> BooleanChunked
where
    F: Fn(&T::Native) -> bool,
{
    let chunks = ca.downcast_iter().map(|arr| {
        let values = arr.values();
        let lower_len = values.partition_point(&is_lower);
        let mut mask = MutableBitmap::with_capacity(arr.len());
        mask.extend_constant(lower_len, lower_part);
        mask.extend_constant(arr.len() - lower_len, !lower_part);
        BooleanArray::from_data_default(mask.into(), None)
    });

    let output_order = if lower_part {
        IsSorted::Descending
    } else {
        IsSorted::Ascending
    };
    let mut ca = BooleanChunked::from_chunk_iter(ca.name(), chunks);
    ca.set_sorted_flag(output_order);
    ca
}

impl<T, Rhs> ChunkCompare<Rhs> for ChunkedArray<T>
where
    T: PolarsNumericType,
    Rhs: ToPrimitive,
{
    type Item = BooleanChunked;
    fn equal(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::eq_scalar(l, rhs))
    }

    fn equal_missing(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::eq_scalar_and_validity(l, rhs))
    }

    fn not_equal(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::neq_scalar(l, rhs))
    }

    fn not_equal_missing(&self, rhs: Rhs) -> BooleanChunked {
        self.primitive_compare_scalar(rhs, |l, rhs| comparison::neq_scalar_and_validity(l, rhs))
    }

    fn gt(&self, rhs: Rhs) -> BooleanChunked {
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                let rhs: T::Native = NumCast::from(rhs).unwrap();
                partition_mask(self, false, |x| x.tot_le(&rhs))
            },
            (IsSorted::Descending, 0) => {
                let rhs: T::Native = NumCast::from(rhs).unwrap();
                partition_mask(self, true, |x| x.tot_gt(&rhs))
            },
            _ => self.primitive_compare_scalar(rhs, |l, rhs| comparison::gt_scalar(l, rhs)),
        }
    }

    fn gt_eq(&self, rhs: Rhs) -> BooleanChunked {
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                let rhs: T::Native = NumCast::from(rhs).unwrap();
                partition_mask(self, false, |x| x.tot_lt(&rhs))
            },
            (IsSorted::Descending, 0) => {
                let rhs: T::Native = NumCast::from(rhs).unwrap();
                partition_mask(self, true, |x| x.tot_ge(&rhs))
            },
            _ => self.primitive_compare_scalar(rhs, |l, rhs| comparison::gt_eq_scalar(l, rhs)),
        }
    }

    fn lt(&self, rhs: Rhs) -> BooleanChunked {
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                let rhs: T::Native = NumCast::from(rhs).unwrap();
                partition_mask(self, true, |x| x.tot_lt(&rhs))
            },
            (IsSorted::Descending, 0) => {
                let rhs: T::Native = NumCast::from(rhs).unwrap();
                partition_mask(self, false, |x| x.tot_ge(&rhs))
            },
            _ => self.primitive_compare_scalar(rhs, |l, rhs| comparison::lt_scalar(l, rhs)),
        }
    }

    fn lt_eq(&self, rhs: Rhs) -> BooleanChunked {
        match (self.is_sorted_flag(), self.null_count()) {
            (IsSorted::Ascending, 0) => {
                let rhs: T::Native = NumCast::from(rhs).unwrap();
                partition_mask(self, true, |x| x.tot_le(&rhs))
            },
            (IsSorted::Descending, 0) => {
                let rhs: T::Native = NumCast::from(rhs).unwrap();
                partition_mask(self, false, |x| x.tot_gt(&rhs))
            },
            _ => self.primitive_compare_scalar(rhs, |l, rhs| comparison::lt_eq_scalar(l, rhs)),
        }
    }
}

impl Utf8Chunked {
    pub(super) fn utf8_compare_scalar(
        &self,
        rhs: &str,
        f: impl Fn(&Utf8Array<i64>, &dyn Scalar) -> BooleanArray,
    ) -> BooleanChunked {
        let scalar = Utf8Scalar::<i64>::new(Some(rhs));
        self.apply_kernel_cast(&|arr| Box::new(f(arr, &scalar)))
    }
}

impl BinaryChunked {
    fn binary_compare_scalar(
        &self,
        rhs: &[u8],
        f: impl Fn(&BinaryArray<i64>, &dyn Scalar) -> BooleanArray,
    ) -> BooleanChunked {
        let scalar = BinaryScalar::<i64>::new(Some(rhs));
        self.apply_kernel_cast(&|arr| Box::new(f(arr, &scalar)))
    }
}

impl ChunkCompare<&[u8]> for BinaryChunked {
    type Item = BooleanChunked;
    fn equal(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::eq_scalar(l, rhs))
    }

    fn equal_missing(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::eq_scalar_and_validity(l, rhs))
    }

    fn not_equal(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::neq_scalar(l, rhs))
    }

    fn not_equal_missing(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::neq_scalar_and_validity(l, rhs))
    }

    fn gt(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::gt_scalar(l, rhs))
    }

    fn gt_eq(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::gt_eq_scalar(l, rhs))
    }

    fn lt(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::lt_scalar(l, rhs))
    }

    fn lt_eq(&self, rhs: &[u8]) -> BooleanChunked {
        self.binary_compare_scalar(rhs, |l, rhs| comparison::lt_eq_scalar(l, rhs))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_binary_search_cmp() {
        let mut s = Series::new("", &[1, 1, 2, 2, 4, 8]);
        s.set_sorted_flag(IsSorted::Ascending);
        let out = s.gt(10).unwrap();
        assert!(!out.any());

        let out = s.gt(0).unwrap();
        assert!(out.all());

        let out = s.gt(2).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [false, false, false, false, true, true])
        );
        let out = s.gt(3).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [false, false, false, false, true, true])
        );

        let out = s.gt_eq(10).unwrap();
        assert!(!out.any());
        let out = s.gt_eq(0).unwrap();
        assert!(out.all());

        let out = s.gt_eq(2).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [false, false, true, true, true, true])
        );
        let out = s.gt_eq(3).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [false, false, false, false, true, true])
        );

        let out = s.lt(10).unwrap();
        assert!(out.all());
        let out = s.lt(0).unwrap();
        assert!(!out.any());

        let out = s.lt(2).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [true, true, false, false, false, false])
        );
        let out = s.lt(3).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [true, true, true, true, false, false])
        );

        let out = s.lt_eq(10).unwrap();
        assert!(out.all());
        let out = s.lt_eq(0).unwrap();
        assert!(!out.any());

        let out = s.lt_eq(2).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [true, true, true, true, false, false])
        );
        let out = s.lt(3).unwrap();
        assert_eq!(
            out.into_series(),
            Series::new("", [true, true, true, true, false, false])
        );
    }
}
