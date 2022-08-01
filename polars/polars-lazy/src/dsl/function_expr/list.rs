use super::*;

#[cfg(feature = "is_in")]
pub(super) fn contains(args: &mut [Series]) -> Result<Series> {
    let list = &args[0];
    let is_in = &args[1];

    is_in.is_in(list).map(|mut ca| {
        ca.rename(list.name());
        ca.into_series()
    })
}
