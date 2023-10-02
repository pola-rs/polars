use std::option::Option;
use std::vec::Vec;
use polars_core::prelude::*;

pub fn list_cum_concat(a: &ListChunked) -> PolarsResult<ListChunked> {

    let mut values: Vec<Option<Series>> = Vec::new();
    let mut acc: Option<Series> = None;
    for value in a.into_iter() {
        match (&mut acc, &value) {
            (Some(acc_inner), Some(v)) => {
                println!("with_state {}", v);
                acc_inner.append(&v);
                values.push(Some(acc_inner.clone()));
            },
            (None, Some(v)) => {
                println!("no_state {}", v);
                acc = Some(v.clone());
                values.push(Some(v.clone()));
            },
            (_, None) => {
                println!("nothing at all");
                values.push(None);
            },
        }
    }

    let mut ca: ListChunked = values.into_iter().collect();
    ca.rename(a.name());
    Ok(ca)
}
