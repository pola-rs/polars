use arrow::legacy::utils::CustomIterTools;
use polars_core::prelude::*;
use polars_core::{with_match_physical_integer_polars_type, POOL};

fn mode_primitive<T: PolarsDataType>(ca: &ChunkedArray<T>) -> PolarsResult<ChunkedArray<T>>
where
    ChunkedArray<T>: IntoGroupsProxy + ChunkTake<[IdxSize]>,
{
    if ca.is_empty() {
        return Ok(ca.clone());
    }
    let parallel = !POOL.current_thread_has_pending_tasks().unwrap_or(false);
    let groups = ca.group_tuples(parallel, false).unwrap();
    let idx = mode_indices(groups);

    // SAFETY:
    // group indices are in bounds
    Ok(unsafe { ca.take_unchecked(idx.as_slice()) })
}

fn mode_f32(ca: &Float32Chunked) -> PolarsResult<Float32Chunked> {
    let s = ca.apply_as_ints(|v| mode(v).unwrap());
    let ca = s.f32().unwrap().clone();
    Ok(ca)
}

fn mode_64(ca: &Float64Chunked) -> PolarsResult<Float64Chunked> {
    let s = ca.apply_as_ints(|v| mode(v).unwrap());
    let ca = s.f64().unwrap().clone();
    Ok(ca)
}

fn mode_indices(groups: GroupsProxy) -> Vec<IdxSize> {
    match groups {
        GroupsProxy::Idx(groups) => {
            let mut groups = groups.into_iter().collect_trusted::<Vec<_>>();
            groups.sort_unstable_by_key(|k| k.1.len());
            let last = &groups.last().unwrap();
            let max_occur = last.1.len();
            groups
                .iter()
                .rev()
                .take_while(|v| v.1.len() == max_occur)
                .map(|v| v.0)
                .collect()
        },
        GroupsProxy::Slice { groups, .. } => {
            let last = groups.last().unwrap();
            let max_occur = last[1];

            groups
                .iter()
                .rev()
                .take_while(|v| {
                    let len = v[1];
                    len == max_occur
                })
                .map(|v| v[0])
                .collect()
        },
    }
}

pub fn mode(s: &Series) -> PolarsResult<Series> {
    let s_phys = s.to_physical_repr();
    let out = match s_phys.dtype() {
        DataType::Binary => mode_primitive(s_phys.binary().unwrap())?.into_series(),
        DataType::Boolean => mode_primitive(s_phys.bool().unwrap())?.into_series(),
        DataType::Float32 => mode_f32(s_phys.f32().unwrap())?.into_series(),
        DataType::Float64 => mode_64(s_phys.f64().unwrap())?.into_series(),
        DataType::String => mode_primitive(&s_phys.str().unwrap().as_binary())?.into_series(),
        dt if dt.is_integer() => {
            with_match_physical_integer_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s_phys.as_ref().as_ref().as_ref();
                mode_primitive(ca)?.into_series()
            })
        },
        _ => polars_bail!(opq = mode, s.dtype()),
    };
    // # Safety: Casting back into the original from physical representation
    unsafe { out.cast_unchecked(s.dtype()) }
}

#[cfg(test)]
mod test {
    use polars_core::prelude::*;

    use super::{mode, mode_primitive};

    #[test]
    fn mode_test() {
        let ca = Int32Chunked::from_slice("test", &[0, 1, 2, 3, 4, 4, 5, 6, 5, 0]);
        let mut result = mode_primitive(&ca).unwrap().to_vec();
        result.sort_by_key(|a| a.unwrap());
        assert_eq!(&result, &[Some(0), Some(4), Some(5)]);

        let ca = Int32Chunked::from_slice("test", &[1, 1]);
        let mut result = mode_primitive(&ca).unwrap().to_vec();
        result.sort_by_key(|a| a.unwrap());
        assert_eq!(&result, &[Some(1)]);

        let ca = Int32Chunked::from_slice("test", &[]);
        let mut result = mode_primitive(&ca).unwrap().to_vec();
        result.sort_by_key(|a| a.unwrap());
        assert_eq!(result, &[]);

        let ca = Float32Chunked::from_slice("test", &[1.0f32, 2.0, 2.0, 3.0, 3.0, 3.0]);
        let result = mode_primitive(&ca).unwrap().to_vec();
        assert_eq!(result, &[Some(3.0f32)]);

        let ca = StringChunked::from_slice("test", &["test", "test", "test", "another test"]);
        let result = mode_primitive(&ca).unwrap();
        let vec_result4: Vec<Option<&str>> = result.into_iter().collect();
        assert_eq!(vec_result4, &[Some("test")]);

        let mut ca_builder = CategoricalChunkedBuilder::new("test", 5, Default::default());
        ca_builder.append_value("test");
        ca_builder.append_value("test");
        ca_builder.append_value("test2");
        ca_builder.append_value("test2");
        ca_builder.append_value("test2");
        let s = ca_builder.finish().into_series();
        let result = mode(&s).unwrap();
        assert_eq!(result.str_value(0).unwrap(), "test2");
        assert_eq!(result.len(), 1);
    }
}
