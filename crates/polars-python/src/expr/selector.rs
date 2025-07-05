use std::hash::{Hash, Hasher};

use polars::prelude::{
    DataType, DataTypeSelector, Selector, TimeUnit, TimeUnitSet, TimeZone, TimeZoneSet,
};
use polars_plan::dsl;
use pyo3::PyResult;
use pyo3::exceptions::PyTypeError;

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

pub fn parse_datatype_selector(selector: PySelector) -> PyResult<DataTypeSelector> {
    selector.inner.to_dtype_selector().ok_or_else(|| {
        PyTypeError::new_err(format!(
            "expected datatype based expression got '{}'",
            selector.inner
        ))
    })
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
        Selector::ByIndex {
            indices: indices.into(),
            strict,
        }
        .into()
    }

    #[staticmethod]
    fn first(strict: bool) -> Self {
        Selector::ByIndex {
            indices: [0].into(),
            strict,
        }
        .into()
    }

    #[staticmethod]
    fn last(strict: bool) -> Self {
        Selector::ByIndex {
            indices: [-1].into(),
            strict,
        }
        .into()
    }

    #[staticmethod]
    fn matches(pattern: String) -> Self {
        Selector::Matches(pattern.into()).into()
    }

    #[staticmethod]
    fn enum_() -> Self {
        DataTypeSelector::Enum.as_selector().into()
    }

    #[staticmethod]
    fn categorical() -> Self {
        DataTypeSelector::Categorical.as_selector().into()
    }

    #[staticmethod]
    fn nested() -> Self {
        DataTypeSelector::Nested.as_selector().into()
    }

    #[staticmethod]
    fn list(inner_dst: Option<Self>) -> PyResult<Self> {
        let inner_dst = parse_datatype_selector(inner_dst)?;
        DataTypeSelector::List(inner_dst).as_selector().into()
    }

    #[staticmethod]
    fn array(inner_dst: Option<Self>, width: Option<usize>) -> PyResult<Self> {
        let inner_dst = parse_datatype_selector(inner_dst)?;
        DataTypeSelector::Array(inner_dst, width)
            .as_selector()
            .into()
    }

    #[staticmethod]
    fn struct_() -> Self {
        DataTypeSelector::Struct.as_selector().into()
    }

    #[staticmethod]
    fn integer() -> Self {
        DataTypeSelector::Integer.as_selector().into()
    }

    #[staticmethod]
    fn signed_integer() -> Self {
        DataTypeSelector::SignedInteger.as_selector().into()
    }

    #[staticmethod]
    fn unsigned_integer() -> Self {
        DataTypeSelector::UnsignedInteger.as_selector().into()
    }

    #[staticmethod]
    fn float() -> Self {
        DataTypeSelector::Float.as_selector().into()
    }

    #[staticmethod]
    fn decimal() -> Self {
        DataTypeSelector::Decimal.as_selector().into()
    }

    #[staticmethod]
    fn numeric() -> Self {
        DataTypeSelector::Numeric.as_selector().into()
    }

    #[staticmethod]
    fn temporal() -> Self {
        DataTypeSelector::Temporal.as_selector().into()
    }

    #[staticmethod]
    fn datetime(tu: Vec<Wrap<TimeUnit>>, tz: Vec<Wrap<Option<TimeZone>>>) -> Self {
        let mut any_of: Option<Vec<TimeZone>> = None;
        let mut tzs = TimeZoneSet {
            allow_unset: false,
            allow_set: true,
            any_of: None,
        };

        let tu = parse_time_unit_set(tu);
        for t in tz {
            let t = t.0;
            match t {
                None => tzs.allow_unset = true,
                Some(s) if s.as_str() == "*" => tzs.allow_set = true,
                Some(t) => any_of.get_or_insert_default().push(t),
            }
        }
        tzs.any_of = any_of.into();
        DataTypeSelector::Datetime(tu, tzs).as_selector().into()
    }

    #[staticmethod]
    fn duration(tu: Vec<Wrap<TimeUnit>>) -> Self {
        let tu = parse_time_unit_set(tu);
        DataTypeSelector::Duration(tu).as_selector().into()
    }

    #[staticmethod]
    fn object() -> Self {
        DataTypeSelector::Object.as_selector().into()
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
