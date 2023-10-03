use std::option::Option;
use std::vec::Vec;

use polars_core::prelude::*;

pub fn list_cum_concat(a: &ListChunked) -> PolarsResult<ListChunked> {
    let mut values: Vec<Option<Series>> = Vec::new();
    let mut acc: Option<Series> = None;
    for value in a {
        match (&mut acc, &value) {
            (Some(acc_inner), Some(v)) => {
                acc_inner.append(v)?;
                values.push(Some(acc_inner.clone()));
            },
            (None, Some(v)) => {
                acc = Some(v.clone());
                values.push(Some(v.clone()));
            },
            (_, None) => {
                values.push(None);
            },
        }
    }

    let mut ca: ListChunked = values.into_iter().collect();
    ca.rename(a.name());
    Ok(ca)
}

pub fn list_cum_set_union(a: &ListChunked) -> PolarsResult<ListChunked> {
    let mut values: Vec<Option<Series>> = Vec::new();
    let mut acc: Option<Series> = None;
    for value in a {
        match (&mut acc, &value) {
            (Some(acc_inner), Some(v)) => {
                let mut new_acc = acc_inner.clone();
                new_acc.append(v)?;
                new_acc = new_acc.0.unique()?;
                acc = Some(new_acc.clone());
                values.push(Some(new_acc));
            },
            (None, Some(v)) => {
                acc = Some(v.0.unique()?);
                values.push(acc.clone());
            },
            (_, None) => {
                values.push(None);
            },
        }
    }

    let mut ca: ListChunked = values.into_iter().collect();
    ca.rename(a.name());
    Ok(ca)
}
