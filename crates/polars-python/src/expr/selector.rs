use std::hash::{Hash, Hasher};

use polars::prelude::{DataType, Selector, TimeUnit, TimeZone, TimeUnitSet};
use polars_plan::dsl;

use crate::prelude::Wrap;

#[pyo3::pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PySelector {
    pub inner: Selector,
}

impl From<Selector> for PySelector {
    fn from(inner: Selector) -> Self {
        Self { inner }
    }
}

fn parse_time_unit_set(time_units: Vec<Wrap<TimeUnit>>) -> TimeUnitSet {
    let mut tu = TimeUnitSet::empty();
    for v in time_units {
        match v.0 {
            TimeUnit::Nanoseconds => tu |= TimeUnitSet::NANO_SECONDS,
            TimeUnit::Microseconds => tu |= TimeUnitSet::MICRO_SECONDS,
            TimeUnit::Milliseconds => tu |= TimeUnitSet::MILLI_SECONDS,
        }
    }
    tu
}

#[cfg(feature = "pymethods")]
#[pyo3::pymethods]
impl PySelector {
    fn union(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() | other.inner.clone(),
        }
    }

    fn difference(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() - other.inner.clone(),
        }
    }

    fn exclusive_or(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() ^ other.inner.clone(),
        }
    }

    fn intersect(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.clone() & other.inner.clone(),
        }
    }

    fn exclude_columns(&self, names: Vec<String>) -> Self {
        self.inner.clone().exclude_cols(names).into()
    }

    fn exclude_dtype(&self, dtypes: Vec<Wrap<DataType>>) -> Self {
        let dtypes = dtypes.into_iter().map(|x| x.0).collect::<Vec<_>>();
        self.inner.clone().exclude_dtype(dtypes).into()
    }

    #[staticmethod]
    fn by_dtype(dtypes: Vec<Wrap<DataType>>) -> Self {
        let dtypes = dtypes.into_iter().map(|x| x.0).collect::<Vec<_>>();
        dsl::dtype_cols(dtypes).into()
    }

    #[staticmethod]
    fn by_name(names: Vec<String>, strict: bool) -> Self {
        let names = names.into_iter().map(Into::into).collect();
        Selector::ByName { names, strict }.into()
    }

    #[staticmethod]
    fn by_index(indices: Vec<i64>, strict: bool) -> Self {
        Selector::ByIndex { indices: indices.into(), strict }.into()
    }

    #[staticmethod]
    fn first(strict: bool) -> Self {
        Selector::ByIndex { indices: [0].into(), strict }.into()
    }

    #[staticmethod]
    fn last(strict: bool) -> Self {
        Selector::ByIndex { indices: [-1].into(), strict }.into()
    }

    #[staticmethod]
    fn matches(pattern: String) -> Self {
        Selector::Matches(pattern.into()).into()
    }

    #[staticmethod]
    fn categorical() -> Self {
        Selector::Categorical.into()
    }

    #[staticmethod]
    fn integer() -> Self {
        Selector::Integer.into()
    }
    
    #[staticmethod]
    fn signed_integer() -> Self {
        Selector::SignedInteger.into()
    }

    #[staticmethod]
    fn unsigned_integer() -> Self {
        Selector::UnsignedInteger.into()
    }

    #[staticmethod]
    fn float() -> Self {
        Selector::Float.into()
    }

    #[staticmethod]
    fn decimal() -> Self {
        Selector::Decimal.into()
    }

    #[staticmethod]
    fn numeric() -> Self {
        Selector::Numeric.into()
    }

    #[staticmethod]
    fn temporal() -> Self {
        Selector::Temporal.into()
    }

    #[staticmethod]
    fn datetime(tu: Vec<Wrap<TimeUnit>>, tz: Option<Vec<Wrap<Option<TimeZone>>>>) -> Self {
        let tu = parse_time_unit_set(tu);
        let tz = tz.map(|v| v.into_iter().map(|v| v.0).collect());
        Selector::Datetime(tu, tz).into()
    }

    #[staticmethod]
    fn duration(tu: Vec<Wrap<TimeUnit>>) -> Self {
        let tu = parse_time_unit_set(tu);
        Selector::Duration(tu).into()
    }

    #[staticmethod]
    fn object() -> Self {
        Selector::Object.into()
    }

    #[staticmethod]
    fn empty() -> Self {
        dsl::empty().into()
    }

    #[staticmethod]
    fn all() -> Self {
        dsl::all().into()
    }

    fn hash(&self) -> u64 {
        let mut hasher = std::hash::DefaultHasher::default();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}
