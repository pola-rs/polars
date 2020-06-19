use super::hash_join::prepare_hashed_relation;
use crate::prelude::*;
use fnv::FnvHashMap;
use std::hash::Hash;

pub struct GroupBy<'a> {
    df: &'a DataFrame,
    // [first idx, [other idx]]
    groups: Vec<(usize, Vec<usize>)>,
}

fn group_by<T>(a: impl Iterator<Item = Option<T>>) -> Vec<(usize, Vec<usize>)>
where
    T: Hash + Eq + Copy,
{
    let hash_tbl = prepare_hashed_relation(a);

    hash_tbl
        .into_iter()
        .map(|(_, indexes)| {
            let first = unsafe { *indexes.get_unchecked(0) };
            (first, indexes)
        })
        .collect()
}

impl DataFrame {
    pub fn group_by(&self, by: &str) -> Option<GroupBy> {
        let groups = if let Some(s) = self.select(by) {
            match s {
                Series::UInt32(ca) => group_by(ca.iter()),
                Series::Int32(ca) => group_by(ca.iter()),
                Series::Int64(ca) => group_by(ca.iter()),
                Series::Bool(ca) => group_by(ca.iter()),
                Series::Utf8(ca) => group_by(ca.iter().map(|v| Some(v))),
                Series::Date32(ca) => group_by(ca.iter().map(|v| Some(v))),
                Series::Date64(ca) => group_by(ca.iter().map(|v| Some(v))),
                Series::Time64Ns(ca) => group_by(ca.iter().map(|v| Some(v))),
                Series::DurationNs(ca) => group_by(ca.iter().map(|v| Some(v))),
                _ => unimplemented!(),
            }
        } else {
            return None;
        };

        Some(GroupBy { df: self, groups })
    }
}
