use polars_arrow::utils::CustomIterTools;

use super::*;

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ListFunction {
    Concat,
    #[cfg(feature = "is_in")]
    Contains,
    Slice,
}

impl Display for ListFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use ListFunction::*;

        let name = match self {
            Concat => "concat",
            #[cfg(feature = "is_in")]
            Contains => "contains",
            Slice => "slice",
        };
        write!(f, "{}", name)
    }
}

#[cfg(feature = "is_in")]
pub(super) fn contains(args: &mut [Series]) -> PolarsResult<Series> {
    let list = &args[0];
    let is_in = &args[1];

    is_in.is_in(list).map(|mut ca| {
        ca.rename(list.name());
        ca.into_series()
    })
}

pub(super) fn slice(args: &mut [Series]) -> PolarsResult<Series> {
    let s = &args[0];
    let list_ca = s.list()?;
    let offset_s = &args[1];
    let length_s = &args[2];

    let mut out: ListChunked = match (offset_s.len(), length_s.len()) {
        (1, 1) => {
            let offset = offset_s.get(0).try_extract::<i64>()?;
            let slice_len = length_s.get(0).try_extract::<usize>()?;
            return Ok(list_ca.lst_slice(offset, slice_len).into_series());
        }
        (1, length_slice_len) => {
            if length_slice_len != list_ca.len() {
                return Err(PolarsError::ComputeError("the length of the slice 'length' argument does not match that of the list column".into()));
            }
            let offset = offset_s.get(0).try_extract::<i64>()?;
            // cast to i64 as it is more likely that it is that dtype
            // instead of usize/u64 (we never need that max length)
            let length_ca = length_s.cast(&DataType::Int64)?;
            let length_ca = length_ca.i64().unwrap();

            list_ca
                .amortized_iter()
                .zip(length_ca.into_iter())
                .map(|(opt_s, opt_length)| match (opt_s, opt_length) {
                    (Some(s), Some(length)) => Some(s.as_ref().slice(offset, length as usize)),
                    _ => None,
                })
                .collect_trusted()
        }
        (offset_len, 1) => {
            if offset_len != list_ca.len() {
                return Err(PolarsError::ComputeError("the length of the slice 'offset' argument does not match that of the list column".into()));
            }
            let length_slice = length_s.get(0).try_extract::<usize>()?;
            let offset_ca = offset_s.cast(&DataType::Int64)?;
            let offset_ca = offset_ca.i64().unwrap();
            list_ca
                .amortized_iter()
                .zip(offset_ca)
                .map(|(opt_s, opt_offset)| match (opt_s, opt_offset) {
                    (Some(s), Some(offset)) => Some(s.as_ref().slice(offset, length_slice)),
                    _ => None,
                })
                .collect_trusted()
        }
        _ => {
            if offset_s.len() != list_ca.len() {
                return Err(PolarsError::ComputeError("the length of the slice 'offset' argument does not match that of the list column".into()));
            }
            if length_s.len() != list_ca.len() {
                return Err(PolarsError::ComputeError("the length of the slice 'length' argument does not match that of the list column".into()));
            }
            let offset_ca = offset_s.cast(&DataType::Int64)?;
            let offset_ca = offset_ca.i64()?;
            // cast to i64 as it is more likely that it is that dtype
            // instead of usize/u64 (we never need that max length)
            let length_ca = length_s.cast(&DataType::Int64)?;
            let length_ca = length_ca.i64().unwrap();

            list_ca
                .amortized_iter()
                .zip(offset_ca.into_iter())
                .zip(length_ca.into_iter())
                .map(
                    |((opt_s, opt_offset), opt_length)| match (opt_s, opt_offset, opt_length) {
                        (Some(s), Some(offset), Some(length)) => {
                            Some(s.as_ref().slice(offset, length as usize))
                        }
                        _ => None,
                    },
                )
                .collect_trusted()
        }
    };
    out.rename(s.name());
    Ok(out.into_series())
}

pub(super) fn concat(s: &mut [Series]) -> PolarsResult<Series> {
    let mut first = std::mem::take(&mut s[0]);
    let other = &s[1..];

    let first_ca = match first.list().ok() {
        Some(ca) => ca,
        None => {
            first = first.reshape(&[-1, 1]).unwrap();
            first.list().unwrap()
        }
    };
    first_ca.lst_concat(other).map(|ca| ca.into_series())
}
