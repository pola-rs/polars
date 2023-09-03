//! Comparison operations on Series.

#[cfg(feature = "dtype-struct")]
use std::ops::Deref;

use super::Series;
use crate::apply_method_physical_numeric;
use crate::prelude::*;
use crate::series::arithmetic::coerce_lhs_rhs;

macro_rules! impl_compare {
    ($self:expr, $rhs:expr, $method:ident) => {{
        let (lhs, rhs) = coerce_lhs_rhs($self, $rhs).expect("cannot coerce datatypes");
        let lhs = lhs.to_physical_repr();
        let rhs = rhs.to_physical_repr();
        match lhs.dtype() {
            DataType::Boolean => lhs.bool().unwrap().$method(rhs.bool().unwrap()),
            DataType::Utf8 => lhs.utf8().unwrap().$method(rhs.utf8().unwrap()),
            DataType::Binary => lhs.binary().unwrap().$method(rhs.binary().unwrap()),
            DataType::UInt8 => lhs.u8().unwrap().$method(rhs.u8().unwrap()),
            DataType::UInt16 => lhs.u16().unwrap().$method(rhs.u16().unwrap()),
            DataType::UInt32 => lhs.u32().unwrap().$method(rhs.u32().unwrap()),
            DataType::UInt64 => lhs.u64().unwrap().$method(rhs.u64().unwrap()),
            DataType::Int8 => lhs.i8().unwrap().$method(rhs.i8().unwrap()),
            DataType::Int16 => lhs.i16().unwrap().$method(rhs.i16().unwrap()),
            DataType::Int32 => lhs.i32().unwrap().$method(rhs.i32().unwrap()),
            DataType::Int64 => lhs.i64().unwrap().$method(rhs.i64().unwrap()),
            DataType::Float32 => lhs.f32().unwrap().$method(rhs.f32().unwrap()),
            DataType::Float64 => lhs.f64().unwrap().$method(rhs.f64().unwrap()),
            DataType::List(_) => lhs.list().unwrap().$method(rhs.list().unwrap()),
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => lhs
                .struct_()
                .unwrap()
                .$method(rhs.struct_().unwrap().deref()),

            _ => unimplemented!(),
        }
    }};
}

#[cfg(feature = "dtype-categorical")]
fn compare_cat_to_str_value<Compare>(
    cat: &Series,
    value: &str,
    name: &str,
    compare: Compare,
    fill_value: bool,
) -> PolarsResult<BooleanChunked>
where
    Compare: Fn(&Series, u32) -> PolarsResult<BooleanChunked>,
{
    let cat = cat.categorical().expect("should be categorical");
    let cat_map = cat.get_rev_map();
    match cat_map.find(value) {
        None => Ok(BooleanChunked::full(name, fill_value, cat.len())),
        Some(cat_idx) => {
            let cat = cat.cast(&DataType::UInt32).unwrap();
            compare(&cat, cat_idx)
        },
    }
}

#[cfg(feature = "dtype-categorical")]
fn compare_cat_to_str_series<Compare>(
    cat: &Series,
    string: &Series,
    name: &str,
    compare: Compare,
    fill_value: bool,
) -> PolarsResult<BooleanChunked>
where
    Compare: Fn(&Series, u32) -> PolarsResult<BooleanChunked>,
{
    match string.utf8()?.get(0) {
        None => Ok(cat.is_null()),
        Some(value) => compare_cat_to_str_value(cat, value, name, compare, fill_value),
    }
}

fn validate_types(left: &DataType, right: &DataType) -> PolarsResult<()> {
    use DataType::*;
    #[cfg(feature = "dtype-categorical")]
    {
        let mismatch = matches!(left, Utf8 | Categorical(_)) && right.is_numeric()
            || left.is_numeric() && matches!(right, Utf8 | Categorical(_));
        polars_ensure!(!mismatch, ComputeError: "cannot compare utf-8 with numeric data");
    }
    #[cfg(not(feature = "dtype-categorical"))]
    {
        let mismatch = matches!(left, Utf8) && right.is_numeric()
            || left.is_numeric() && matches!(right, Utf8);
        polars_ensure!(!mismatch, ComputeError: "cannot compare utf-8 with numeric data");
    }
    Ok(())
}

impl ChunkCompare<&Series> for Series {
    type Item = PolarsResult<BooleanChunked>;

    /// Create a boolean mask by checking for equality.
    fn equal(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), rhs.dtype())?;
        use DataType::*;
        let mut out = match (self.dtype(), rhs.dtype(), self.len(), rhs.len()) {
            #[cfg(feature = "dtype-categorical")]
            (Categorical(_), Utf8, _, 1) => {
                return compare_cat_to_str_series(
                    self,
                    rhs,
                    self.name(),
                    |s, idx| s.equal(idx),
                    false,
                );
            },
            #[cfg(feature = "dtype-categorical")]
            (Utf8, Categorical(_), 1, _) => {
                return compare_cat_to_str_series(
                    rhs,
                    self,
                    self.name(),
                    |s, idx| s.equal(idx),
                    false,
                );
            },
            #[cfg(feature = "dtype-categorical")]
            (Categorical(Some(rev_map_l)), Categorical(Some(rev_map_r)), _, _) => {
                if rev_map_l.same_src(rev_map_r) {
                    let rhs = rhs.categorical().unwrap().logical();

                    // first check the rev-map
                    if rhs.len() == 1 && rhs.null_count() == 0 {
                        let rhs = rhs.get(0).unwrap();
                        if rev_map_l.get_optional(rhs).is_none() {
                            return Ok(BooleanChunked::full(self.name(), false, self.len()));
                        }
                    }

                    self.categorical().unwrap().logical().equal(rhs)
                } else {
                    polars_bail!(
                        ComputeError:
                        "cannot compare categoricals originating from different sources; \
                        consider setting a global string cache"
                    );
                }
            },
            (Null, Null, _, _) => BooleanChunked::full_null(self.name(), self.len()),
            _ => {
                impl_compare!(self, rhs, equal)
            },
        };
        out.rename(self.name());
        Ok(out)
    }

    /// Create a boolean mask by checking for equality.
    fn equal_missing(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), rhs.dtype())?;
        use DataType::*;
        let mut out = match (self.dtype(), rhs.dtype(), self.len(), rhs.len()) {
            #[cfg(feature = "dtype-categorical")]
            (Categorical(_), Utf8, _, 1) => {
                return compare_cat_to_str_series(
                    self,
                    rhs,
                    self.name(),
                    |s, idx| s.equal_missing(idx),
                    false,
                );
            },
            #[cfg(feature = "dtype-categorical")]
            (Utf8, Categorical(_), 1, _) => {
                return compare_cat_to_str_series(
                    rhs,
                    self,
                    self.name(),
                    |s, idx| s.equal_missing(idx),
                    false,
                );
            },
            #[cfg(feature = "dtype-categorical")]
            (Categorical(Some(rev_map_l)), Categorical(Some(rev_map_r)), _, _) => {
                if rev_map_l.same_src(rev_map_r) {
                    let rhs = rhs.categorical().unwrap().logical();

                    // first check the rev-map
                    if rhs.len() == 1 && rhs.null_count() == 0 {
                        let rhs = rhs.get(0).unwrap();
                        if rev_map_l.get_optional(rhs).is_none() {
                            return Ok(BooleanChunked::full(self.name(), false, self.len()));
                        }
                    }

                    self.categorical().unwrap().logical().equal_missing(rhs)
                } else {
                    polars_bail!(
                        ComputeError:
                        "cannot compare categoricals originating from different sources; \
                        consider setting a global string cache"
                    );
                }
            },
            (Null, Null, _, _) => BooleanChunked::full(self.name(), true, self.len()),
            _ => {
                impl_compare!(self, rhs, equal_missing)
            },
        };
        out.rename(self.name());
        Ok(out)
    }

    /// Create a boolean mask by checking for inequality.
    fn not_equal(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), rhs.dtype())?;
        use DataType::*;
        let mut out = match (self.dtype(), rhs.dtype(), self.len(), rhs.len()) {
            #[cfg(feature = "dtype-categorical")]
            (Categorical(_), Utf8, _, 1) => {
                return compare_cat_to_str_series(
                    self,
                    rhs,
                    self.name(),
                    |s, idx| s.not_equal(idx),
                    true,
                );
            },
            #[cfg(feature = "dtype-categorical")]
            (Utf8, Categorical(_), 1, _) => {
                return compare_cat_to_str_series(
                    rhs,
                    self,
                    self.name(),
                    |s, idx| s.not_equal(idx),
                    true,
                );
            },
            #[cfg(feature = "dtype-categorical")]
            (Categorical(Some(rev_map_l)), Categorical(Some(rev_map_r)), _, _) => {
                if rev_map_l.same_src(rev_map_r) {
                    let rhs = rhs.categorical().unwrap().logical();

                    // first check the rev-map
                    if rhs.len() == 1 && rhs.null_count() == 0 {
                        let rhs = rhs.get(0).unwrap();
                        if rev_map_l.get_optional(rhs).is_none() {
                            return Ok(BooleanChunked::full(self.name(), true, self.len()));
                        }
                    }

                    self.categorical().unwrap().logical().not_equal(rhs)
                } else {
                    polars_bail!(
                        ComputeError:
                        "cannot compare categoricals originating from different sources; \
                        consider setting a global string cache"
                    );
                }
            },
            (Null, Null, _, _) => BooleanChunked::full_null(self.name(), self.len()),
            _ => {
                impl_compare!(self, rhs, not_equal)
            },
        };
        out.rename(self.name());
        Ok(out)
    }

    /// Create a boolean mask by checking for inequality.
    fn not_equal_missing(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), rhs.dtype())?;
        use DataType::*;
        let mut out = match (self.dtype(), rhs.dtype(), self.len(), rhs.len()) {
            #[cfg(feature = "dtype-categorical")]
            (Categorical(_), Utf8, _, 1) => {
                return compare_cat_to_str_series(
                    self,
                    rhs,
                    self.name(),
                    |s, idx| s.not_equal_missing(idx),
                    true,
                );
            },
            #[cfg(feature = "dtype-categorical")]
            (Utf8, Categorical(_), 1, _) => {
                return compare_cat_to_str_series(
                    rhs,
                    self,
                    self.name(),
                    |s, idx| s.not_equal_missing(idx),
                    true,
                );
            },
            #[cfg(feature = "dtype-categorical")]
            (Categorical(Some(rev_map_l)), Categorical(Some(rev_map_r)), _, _) => {
                if rev_map_l.same_src(rev_map_r) {
                    let rhs = rhs.categorical().unwrap().logical();

                    // first check the rev-map
                    if rhs.len() == 1 && rhs.null_count() == 0 {
                        let rhs = rhs.get(0).unwrap();
                        if rev_map_l.get_optional(rhs).is_none() {
                            return Ok(BooleanChunked::full(self.name(), true, self.len()));
                        }
                    }

                    self.categorical().unwrap().logical().not_equal_missing(rhs)
                } else {
                    polars_bail!(
                        ComputeError:
                        "cannot compare categoricals originating from different sources; \
                        consider setting a global string cache"
                    );
                }
            },
            (Null, Null, _, _) => BooleanChunked::full(self.name(), false, self.len()),
            _ => {
                impl_compare!(self, rhs, not_equal_missing)
            },
        };
        out.rename(self.name());
        Ok(out)
    }

    /// Create a boolean mask by checking if self > rhs.
    fn gt(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), rhs.dtype())?;
        let mut out = impl_compare!(self, rhs, gt);
        out.rename(self.name());
        Ok(out)
    }

    /// Create a boolean mask by checking if self >= rhs.
    fn gt_eq(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), rhs.dtype())?;
        let mut out = impl_compare!(self, rhs, gt_eq);
        out.rename(self.name());
        Ok(out)
    }

    /// Create a boolean mask by checking if self < rhs.
    fn lt(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), rhs.dtype())?;
        let mut out = impl_compare!(self, rhs, lt);
        out.rename(self.name());
        Ok(out)
    }

    /// Create a boolean mask by checking if self <= rhs.
    fn lt_eq(&self, rhs: &Series) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), rhs.dtype())?;
        let mut out = impl_compare!(self, rhs, lt_eq);
        out.rename(self.name());
        Ok(out)
    }
}

impl<Rhs> ChunkCompare<Rhs> for Series
where
    Rhs: NumericNative,
{
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: Rhs) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, equal, rhs))
    }

    fn equal_missing(&self, rhs: Rhs) -> Self::Item {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, equal_missing, rhs))
    }

    fn not_equal(&self, rhs: Rhs) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, not_equal, rhs))
    }

    fn not_equal_missing(&self, rhs: Rhs) -> Self::Item {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, not_equal_missing, rhs))
    }

    fn gt(&self, rhs: Rhs) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, gt, rhs))
    }

    fn gt_eq(&self, rhs: Rhs) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, gt_eq, rhs))
    }

    fn lt(&self, rhs: Rhs) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, lt, rhs))
    }

    fn lt_eq(&self, rhs: Rhs) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Int8)?;
        let s = self.to_physical_repr();
        Ok(apply_method_physical_numeric!(&s, lt_eq, rhs))
    }
}

fn compare_series_str(
    lhs: &Series,
    rhs: &str,
    op: impl Fn(&Utf8Chunked, &str) -> BooleanChunked,
) -> PolarsResult<BooleanChunked> {
    validate_types(lhs.dtype(), &DataType::Utf8)?;
    lhs.utf8().map(|ca| op(ca, rhs)).map_err(|_| {
        polars_err!(
            ComputeError: "cannot compare str value to series of type {}", lhs.dtype(),
        )
    })
}

impl ChunkCompare<&str> for Series {
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: &str) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Utf8)?;
        use DataType::*;
        match self.dtype() {
            Utf8 => Ok(self.utf8().unwrap().equal(rhs)),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_) => {
                compare_cat_to_str_value(self, rhs, self.name(), |lhs, idx| lhs.equal(idx), false)
            },
            _ => Ok(BooleanChunked::full(self.name(), false, self.len())),
        }
    }

    fn equal_missing(&self, rhs: &str) -> Self::Item {
        validate_types(self.dtype(), &DataType::Utf8)?;
        use DataType::*;
        match self.dtype() {
            Utf8 => Ok(self.utf8().unwrap().equal(rhs)),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_) => compare_cat_to_str_value(
                self,
                rhs,
                self.name(),
                |lhs, idx| lhs.equal_missing(idx),
                false,
            ),
            _ => Ok(BooleanChunked::full(self.name(), false, self.len())),
        }
    }

    fn not_equal(&self, rhs: &str) -> PolarsResult<BooleanChunked> {
        validate_types(self.dtype(), &DataType::Utf8)?;
        use DataType::*;
        match self.dtype() {
            Utf8 => Ok(self.utf8().unwrap().not_equal(rhs)),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_) => compare_cat_to_str_value(
                self,
                rhs,
                self.name(),
                |lhs, idx| lhs.not_equal(idx),
                true,
            ),
            _ => Ok(BooleanChunked::full(self.name(), false, self.len())),
        }
    }

    fn not_equal_missing(&self, rhs: &str) -> Self::Item {
        validate_types(self.dtype(), &DataType::Utf8)?;
        use DataType::*;
        match self.dtype() {
            Utf8 => Ok(self.utf8().unwrap().not_equal(rhs)),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_) => compare_cat_to_str_value(
                self,
                rhs,
                self.name(),
                |lhs, idx| lhs.not_equal_missing(idx),
                true,
            ),
            _ => Ok(BooleanChunked::full(self.name(), false, self.len())),
        }
    }

    fn gt(&self, rhs: &str) -> PolarsResult<BooleanChunked> {
        compare_series_str(self, rhs, |lhs, rhs| lhs.gt(rhs))
    }

    fn gt_eq(&self, rhs: &str) -> PolarsResult<BooleanChunked> {
        compare_series_str(self, rhs, |lhs, rhs| lhs.gt_eq(rhs))
    }

    fn lt(&self, rhs: &str) -> PolarsResult<BooleanChunked> {
        compare_series_str(self, rhs, |lhs, rhs| lhs.lt(rhs))
    }

    fn lt_eq(&self, rhs: &str) -> PolarsResult<BooleanChunked> {
        compare_series_str(self, rhs, |lhs, rhs| lhs.lt_eq(rhs))
    }
}
