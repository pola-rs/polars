use crate::prelude::*;

pub(crate) fn to_physical(s: &[Series]) -> Vec<Series> {
    s.iter()
        .map(|s| s.to_physical_repr().into_owned())
        .collect()
}
