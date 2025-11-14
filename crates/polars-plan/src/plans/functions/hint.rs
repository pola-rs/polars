use std::fmt;
use std::sync::Arc;

use polars_core::prelude::PlHashSet;
use polars_utils::pl_str::PlSmallStr;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Hash)]
pub struct Sorted {
    pub column: PlSmallStr,
    pub descending: bool,
    pub nulls_last: bool,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Hash, strum_macros::IntoStaticStr)]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum HintIR {
    Sorted(Arc<[Sorted]>),
}

impl HintIR {
    pub fn project(&self, projected_names: &PlHashSet<PlSmallStr>) -> Option<HintIR> {
        match self {
            Self::Sorted(s) => {
                let num_matches = s
                    .iter()
                    .filter(|i| projected_names.contains(&i.column))
                    .count();

                if num_matches == s.len() {
                    return Some(Self::Sorted(s.clone()));
                } else if num_matches == 0 {
                    return None;
                }

                let mut sorted = Vec::with_capacity(num_matches);
                sorted.extend(
                    s.iter()
                        .filter(|i| projected_names.contains(&i.column))
                        .cloned(),
                );
                Some(Self::Sorted(sorted.into()))
            },
        }
    }
}

impl fmt::Display for Sorted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "'{}': {{ descending: {}, nulls_last: {} }}",
            self.column, self.descending, self.nulls_last
        )
    }
}

impl fmt::Debug for HintIR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl fmt::Display for HintIR {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HintIR::Sorted(s) => {
                write!(f, "sorted(")?;
                if let Some(fst) = s.first() {
                    fst.fmt(f)?;
                    for si in &s[1..] {
                        f.write_str(", ")?;
                        si.fmt(f)?;
                    }
                }
                write!(f, ")")
            },
        }
    }
}
