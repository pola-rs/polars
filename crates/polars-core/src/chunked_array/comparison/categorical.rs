use crate::prelude::arity::unary_mut_values;
use crate::prelude::*;

fn str_to_cat_enum(map: &CategoricalMapping, s: &str) -> PolarsResult<CatSize> {
    map.get_cat(s).ok_or_else(|| polars_err!(InvalidOperation: "conversion from `str` to `enum` failed for value \"{s}\""))
}

fn cat_equality_helper<T: PolarsCategoricalType, EqPhys>(
    lhs: &CategoricalChunked<T>,
    rhs: &CategoricalChunked<T>,
    eq_phys: EqPhys,
) -> PolarsResult<BooleanChunked>
where
    EqPhys:
        Fn(&ChunkedArray<T::PolarsPhysical>, &ChunkedArray<T::PolarsPhysical>) -> BooleanChunked,
{
    lhs.dtype().matches_schema_type(rhs.dtype())?;
    Ok(eq_phys(lhs.physical(), rhs.physical()))
}

fn cat_compare_helper<T: PolarsCategoricalType, Cmp, CmpPhys>(
    lhs: &CategoricalChunked<T>,
    rhs: &CategoricalChunked<T>,
    cmp: Cmp,
    cmp_phys: CmpPhys,
) -> PolarsResult<BooleanChunked>
where
    Cmp: Fn(&str, &str) -> bool,
    CmpPhys:
        Fn(&ChunkedArray<T::PolarsPhysical>, &ChunkedArray<T::PolarsPhysical>) -> BooleanChunked,
{
    lhs.dtype().matches_schema_type(rhs.dtype())?;
    if lhs.is_enum() {
        return Ok(cmp_phys(lhs.physical(), rhs.physical()));
    }
    let mapping = lhs.get_mapping();
    match (lhs.len(), rhs.len()) {
        (lhs_len, 1) => {
            let Some(cat) = rhs.physical().get(0) else {
                return Ok(BooleanChunked::full_null(lhs.name().clone(), lhs_len));
            };

            // SAFETY: physical is in range of the mapping.
            let v = unsafe { mapping.cat_to_str_unchecked(cat.as_cat()) };
            Ok(lhs
                .iter_str()
                .map(|opt_s| opt_s.map(|s| cmp(s, v)))
                .collect_ca_trusted(lhs.name().clone()))
        },
        (1, rhs_len) => {
            let Some(cat) = lhs.physical().get(0) else {
                return Ok(BooleanChunked::full_null(lhs.name().clone(), rhs_len));
            };

            // SAFETY: physical is in range of the mapping.
            let v = unsafe { mapping.cat_to_str_unchecked(cat.as_cat()) };
            Ok(rhs
                .iter_str()
                .map(|opt_s| opt_s.map(|s| cmp(v, s)))
                .collect_ca_trusted(lhs.name().clone()))
        },
        (lhs_len, rhs_len) => {
            assert!(lhs_len == rhs_len);
            Ok(lhs
                .iter_str()
                .zip(rhs.iter_str())
                .map(|(l, r)| match (l, r) {
                    (None, _) => None,
                    (_, None) => None,
                    (Some(l), Some(r)) => Some(cmp(l, r)),
                })
                .collect_ca_trusted(lhs.name().clone()))
        },
    }
}

fn cat_str_equality_helper<T: PolarsCategoricalType, Eq, EqPhysScalar, EqStrScalar>(
    lhs: &CategoricalChunked<T>,
    rhs: &StringChunked,
    eq: Eq,
    eq_phys_scalar: EqPhysScalar,
    eq_str_scalar: EqStrScalar,
) -> BooleanChunked
where
    Eq: Fn(Option<&str>, Option<&str>) -> Option<bool>,
    EqPhysScalar: Fn(&ChunkedArray<T::PolarsPhysical>, T::Native) -> BooleanChunked,
    EqStrScalar: Fn(&StringChunked, &str) -> BooleanChunked,
{
    let mapping = lhs.get_mapping();
    let null_eq = eq(None, None);
    match (lhs.len(), rhs.len()) {
        (lhs_len, 1) => {
            let Some(s) = rhs.get(0) else {
                return match null_eq {
                    Some(true) => lhs.physical().is_null(),
                    Some(false) => lhs.physical().is_not_null(),
                    None => BooleanChunked::full_null(lhs.name().clone(), lhs_len),
                };
            };

            let is_eq = eq(Some(""), Some("")).unwrap();
            cat_str_scalar_equality_helper(lhs, s, is_eq, null_eq.is_some(), &eq_phys_scalar)
        },
        (1, rhs_len) => {
            let Some(cat) = lhs.physical().get(0) else {
                return match null_eq {
                    Some(true) => rhs.is_null().with_name(lhs.name().clone()),
                    Some(false) => rhs.is_not_null().with_name(lhs.name().clone()),
                    None => BooleanChunked::full_null(lhs.name().clone(), rhs_len),
                };
            };

            // SAFETY: physical is in range of the mapping.
            let s = unsafe { mapping.cat_to_str_unchecked(cat.as_cat()) };
            eq_str_scalar(rhs, s).with_name(lhs.name().clone())
        },
        (lhs_len, rhs_len) => {
            assert!(lhs_len == rhs_len);
            lhs.iter_str()
                .zip(rhs.iter())
                .map(|(l, r)| eq(l, r))
                .collect_ca_trusted(lhs.name().clone())
        },
    }
}

fn cat_str_compare_helper<T: PolarsCategoricalType, Cmp, CmpStrScalar>(
    lhs: &CategoricalChunked<T>,
    rhs: &StringChunked,
    cmp: Cmp,
    cmp_str_scalar: CmpStrScalar,
) -> BooleanChunked
where
    Cmp: Fn(&str, &str) -> bool,
    CmpStrScalar: Fn(&str, &StringChunked) -> BooleanChunked,
{
    let mapping = lhs.get_mapping();
    match (lhs.len(), rhs.len()) {
        (lhs_len, 1) => {
            let Some(s) = rhs.get(0) else {
                return BooleanChunked::full_null(lhs.name().clone(), lhs_len);
            };
            cat_str_scalar_compare_helper(lhs, s, cmp)
        },
        (1, rhs_len) => {
            let Some(cat) = lhs.physical().get(0) else {
                return BooleanChunked::full_null(lhs.name().clone(), rhs_len);
            };

            // SAFETY: physical is in range of the mapping.
            let s = unsafe { mapping.cat_to_str_unchecked(cat.as_cat()) };
            cmp_str_scalar(s, rhs).with_name(lhs.name().clone())
        },
        (lhs_len, rhs_len) => {
            assert!(lhs_len == rhs_len);
            lhs.iter_str()
                .zip(rhs.iter())
                .map(|(l, r)| match (l, r) {
                    (None, _) => None,
                    (_, None) => None,
                    (Some(l), Some(r)) => Some(cmp(l, r)),
                })
                .collect_ca_trusted(lhs.name().clone())
        },
    }
}

fn cat_str_phys_compare_helper<T: PolarsCategoricalType, Cmp>(
    lhs: &CategoricalChunked<T>,
    rhs: &StringChunked,
    cmp: Cmp,
) -> PolarsResult<BooleanChunked>
where
    Cmp: Fn(T::Native, T::Native) -> bool,
{
    let mapping = lhs.get_mapping();
    match (lhs.len(), rhs.len()) {
        (lhs_len, 1) => {
            let Some(s) = rhs.get(0) else {
                return Ok(BooleanChunked::full_null(lhs.name().clone(), lhs_len));
            };
            cat_str_scalar_phys_compare_helper(lhs, s, cmp)
        },
        (1, rhs_len) => {
            let Some(cat) = lhs.physical().get(0) else {
                return Ok(BooleanChunked::full_null(lhs.name().clone(), rhs_len));
            };

            rhs.iter()
                .map(|opt_r| {
                    if let Some(r) = opt_r {
                        let r = T::Native::from_cat(str_to_cat_enum(mapping, r)?);
                        Ok(Some(cmp(cat, r)))
                    } else {
                        Ok(None)
                    }
                })
                .try_collect_ca_trusted(lhs.name().clone())
        },
        (lhs_len, rhs_len) => {
            assert!(lhs_len == rhs_len);
            lhs.physical()
                .iter()
                .zip(rhs.iter())
                .map(|(l, r)| match (l, r) {
                    (None, _) => Ok(None),
                    (_, None) => Ok(None),
                    (Some(l), Some(r)) => {
                        let r = T::Native::from_cat(str_to_cat_enum(mapping, r)?);
                        Ok(Some(cmp(l, r)))
                    },
                })
                .try_collect_ca_trusted(lhs.name().clone())
        },
    }
}

fn cat_str_scalar_equality_helper<T: PolarsCategoricalType, EqPhysScalar>(
    lhs: &CategoricalChunked<T>,
    rhs: &str,
    is_eq: bool,
    missing: bool,
    eq_phys_scalar: EqPhysScalar,
) -> BooleanChunked
where
    EqPhysScalar: Fn(&ChunkedArray<T::PolarsPhysical>, T::Native) -> BooleanChunked,
{
    let mapping = lhs.get_mapping();
    let Some(cat) = mapping.get_cat(rhs) else {
        return if missing {
            if is_eq {
                BooleanChunked::full(lhs.name().clone(), false, lhs.len())
            } else {
                BooleanChunked::full(lhs.name().clone(), true, lhs.len())
            }
        } else {
            unary_mut_values(lhs.physical(), |arr| {
                BooleanArray::full(arr.len(), !is_eq, ArrowDataType::Boolean)
            })
        };
    };

    eq_phys_scalar(lhs.physical(), T::Native::from_cat(cat))
}

fn cat_str_scalar_compare_helper<T: PolarsCategoricalType, Cmp>(
    lhs: &CategoricalChunked<T>,
    rhs: &str,
    cmp: Cmp,
) -> BooleanChunked
where
    Cmp: Fn(&str, &str) -> bool,
{
    lhs.iter_str()
        .map(|opt_l| opt_l.map(|l| cmp(l, rhs)))
        .collect_ca_trusted(lhs.name().clone())
}

fn cat_str_scalar_phys_compare_helper<T: PolarsCategoricalType, Cmp>(
    lhs: &CategoricalChunked<T>,
    rhs: &str,
    cmp: Cmp,
) -> PolarsResult<BooleanChunked>
where
    Cmp: Fn(T::Native, T::Native) -> bool,
{
    let r = T::Native::from_cat(str_to_cat_enum(lhs.get_mapping(), rhs)?);
    Ok(lhs
        .physical()
        .iter()
        .map(|opt_l| opt_l.map(|l| cmp(l, r)))
        .collect_ca_trusted(lhs.name().clone()))
}

impl<T: PolarsCategoricalType> ChunkCompareEq<&CategoricalChunked<T>> for CategoricalChunked<T>
where
    ChunkedArray<T::PolarsPhysical>:
        for<'a> ChunkCompareEq<&'a ChunkedArray<T::PolarsPhysical>, Item = BooleanChunked>,
{
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: &Self) -> Self::Item {
        cat_equality_helper(self, rhs, |l, r| l.equal(r))
    }

    fn equal_missing(&self, rhs: &Self) -> Self::Item {
        cat_equality_helper(self, rhs, |l, r| l.equal_missing(r))
    }

    fn not_equal(&self, rhs: &Self) -> Self::Item {
        cat_equality_helper(self, rhs, |l, r| l.not_equal(r))
    }

    fn not_equal_missing(&self, rhs: &Self) -> Self::Item {
        cat_equality_helper(self, rhs, |l, r| l.not_equal_missing(r))
    }
}

impl<T: PolarsCategoricalType> ChunkCompareIneq<&CategoricalChunked<T>> for CategoricalChunked<T>
where
    ChunkedArray<T::PolarsPhysical>:
        for<'a> ChunkCompareIneq<&'a ChunkedArray<T::PolarsPhysical>, Item = BooleanChunked>,
{
    type Item = PolarsResult<BooleanChunked>;

    fn gt(&self, rhs: &CategoricalChunked<T>) -> Self::Item {
        cat_compare_helper(self, rhs, |l, r| l > r, |l, r| l.gt(r))
    }

    fn gt_eq(&self, rhs: &CategoricalChunked<T>) -> Self::Item {
        cat_compare_helper(self, rhs, |l, r| l >= r, |l, r| l.gt_eq(r))
    }

    fn lt(&self, rhs: &CategoricalChunked<T>) -> Self::Item {
        cat_compare_helper(self, rhs, |l, r| l < r, |l, r| l.lt(r))
    }

    fn lt_eq(&self, rhs: &CategoricalChunked<T>) -> Self::Item {
        cat_compare_helper(self, rhs, |l, r| l <= r, |l, r| l.lt_eq(r))
    }
}

impl<T: PolarsCategoricalType> ChunkCompareEq<&StringChunked> for CategoricalChunked<T>
where
    ChunkedArray<T::PolarsPhysical>: for<'a> ChunkCompareEq<T::Native, Item = BooleanChunked>,
{
    type Item = BooleanChunked;

    fn equal(&self, rhs: &StringChunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            |l, r| l.zip(r).map(|(l, r)| l == r),
            |l, c| l.equal(c),
            |r, c| r.equal(c),
        )
    }

    fn equal_missing(&self, rhs: &StringChunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            |l, r| Some(l == r),
            |l, c| l.equal_missing(c),
            |r, c| r.equal_missing(c),
        )
    }

    fn not_equal(&self, rhs: &StringChunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            |l, r| l.zip(r).map(|(l, r)| l != r),
            |l, c| l.not_equal(c),
            |r, c| r.not_equal(c),
        )
    }

    fn not_equal_missing(&self, rhs: &StringChunked) -> Self::Item {
        cat_str_equality_helper(
            self,
            rhs,
            |l, r| Some(l != r),
            |l, c| l.not_equal_missing(c),
            |r, c| r.not_equal_missing(c),
        )
    }
}

impl<T: PolarsCategoricalType> ChunkCompareIneq<&StringChunked> for CategoricalChunked<T> {
    type Item = PolarsResult<BooleanChunked>;

    fn gt(&self, rhs: &StringChunked) -> Self::Item {
        if self.is_enum() {
            cat_str_phys_compare_helper(self, rhs, |l, r| l > r)
        } else {
            Ok(cat_str_compare_helper(
                self,
                rhs,
                |l, r| l > r,
                |c, r| r.lt(c),
            ))
        }
    }

    fn gt_eq(&self, rhs: &StringChunked) -> Self::Item {
        if self.is_enum() {
            cat_str_phys_compare_helper(self, rhs, |l, r| l >= r)
        } else {
            Ok(cat_str_compare_helper(
                self,
                rhs,
                |l, r| l >= r,
                |c, r| r.lt_eq(c),
            ))
        }
    }

    fn lt(&self, rhs: &StringChunked) -> Self::Item {
        if self.is_enum() {
            cat_str_phys_compare_helper(self, rhs, |l, r| l < r)
        } else {
            Ok(cat_str_compare_helper(
                self,
                rhs,
                |l, r| l < r,
                |c, r| r.gt(c),
            ))
        }
    }

    fn lt_eq(&self, rhs: &StringChunked) -> Self::Item {
        if self.is_enum() {
            cat_str_phys_compare_helper(self, rhs, |l, r| l <= r)
        } else {
            Ok(cat_str_compare_helper(
                self,
                rhs,
                |l, r| l <= r,
                |c, r| r.gt_eq(c),
            ))
        }
    }
}

impl<T: PolarsCategoricalType> ChunkCompareEq<&str> for CategoricalChunked<T>
where
    ChunkedArray<T::PolarsPhysical>: for<'a> ChunkCompareEq<T::Native, Item = BooleanChunked>,
{
    type Item = BooleanChunked;

    fn equal(&self, rhs: &str) -> Self::Item {
        cat_str_scalar_equality_helper(self, rhs, true, false, |l, c| l.equal(c))
    }

    fn equal_missing(&self, rhs: &str) -> Self::Item {
        cat_str_scalar_equality_helper(self, rhs, true, true, |l, c| l.equal_missing(c))
    }

    fn not_equal(&self, rhs: &str) -> Self::Item {
        cat_str_scalar_equality_helper(self, rhs, false, false, |r, c| r.not_equal(c))
    }

    fn not_equal_missing(&self, rhs: &str) -> Self::Item {
        cat_str_scalar_equality_helper(self, rhs, false, true, |l, c| l.not_equal_missing(c))
    }
}

impl<T: PolarsCategoricalType> ChunkCompareIneq<&str> for CategoricalChunked<T> {
    type Item = BooleanChunked;

    fn gt(&self, rhs: &str) -> Self::Item {
        cat_str_scalar_compare_helper(self, rhs, |l, r| l > r)
    }

    fn gt_eq(&self, rhs: &str) -> Self::Item {
        cat_str_scalar_compare_helper(self, rhs, |l, r| l >= r)
    }

    fn lt(&self, rhs: &str) -> Self::Item {
        cat_str_scalar_compare_helper(self, rhs, |l, r| l < r)
    }

    fn lt_eq(&self, rhs: &str) -> Self::Item {
        cat_str_scalar_compare_helper(self, rhs, |l, r| l <= r)
    }
}
