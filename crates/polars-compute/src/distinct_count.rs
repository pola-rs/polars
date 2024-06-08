use arrow::array::{Array, BooleanArray};

/// Kernel to calculate the number of unique elements
pub trait DistinctCountKernel {
    /// Calculate the number of unique elements in [`Self`]
    ///
    /// A null is also considered a unique value
    fn distinct_count(&self) -> usize;

    /// Calculate the number of unique non-null elements in [`Self`]
    fn distinct_non_null_count(&self) -> usize;
}

impl DistinctCountKernel for BooleanArray {
    fn distinct_count(&self) -> usize {
        if self.len() == 0 {
            return 0;
        }

        let null_count = self.null_count();

        if self.len() == null_count {
            return 1;
        }

        let values = self.values();

        if null_count == 0 {
            let unset_bits = values.unset_bits();
            let is_uniform = unset_bits == 0 || unset_bits == values.len();
            return 2 - usize::from(is_uniform);
        }

        let validity = self.validity().unwrap();
        let set_bits = values.num_intersections_with(validity);
        let is_uniform = set_bits == 0 || set_bits == validity.set_bits();
        2 + usize::from(!is_uniform)
    }

    #[inline]
    fn distinct_non_null_count(&self) -> usize {
        self.distinct_count() - usize::from(self.null_count() > 0)
    }
}

#[test]
fn test_boolean_distinct_count() {
    use arrow::bitmap::Bitmap;
    use arrow::datatypes::ArrowDataType;

    macro_rules! assert_bool_dc {
        ($values:expr, $validity:expr => $dc:expr) => {
            let validity: Option<Bitmap> =
                <Option<Vec<bool>>>::map($validity, |v| Bitmap::from_iter(v));
            let arr =
                BooleanArray::new(ArrowDataType::Boolean, Bitmap::from_iter($values), validity);
            assert_eq!(arr.distinct_count(), $dc);
        };
    }

    assert_bool_dc!(vec![], None => 0);
    assert_bool_dc!(vec![], Some(vec![]) => 0);
    assert_bool_dc!(vec![true], None => 1);
    assert_bool_dc!(vec![true], Some(vec![true]) => 1);
    assert_bool_dc!(vec![true], Some(vec![false]) => 1);
    assert_bool_dc!(vec![true, false], None => 2);
    assert_bool_dc!(vec![true, false, false], None => 2);
    assert_bool_dc!(vec![true, false, false], Some(vec![true, true, false]) => 3);

    // Copied from https://github.com/pola-rs/polars/pull/16765#discussion_r1629426159
    assert_bool_dc!(vec![true, true, true, true, true], Some(vec![true, false, true, false, false]) => 2);
    assert_bool_dc!(vec![false, true, false, true, true], Some(vec![true, false, true, false, false]) => 2);
    assert_bool_dc!(vec![true, false, true, false, true, true], Some(vec![true, true, false, true, false, false]) => 3);
}
