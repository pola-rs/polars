use polars::prelude::*;
use polars_core::utils::CustomIterTools;
use pyo3::{PyAny, PyResult};

pub fn py_seq_to_list(name: &str, seq: &PyAny, dtype: &DataType) -> PyResult<Series> {
    let len = seq.len()?;
    let s = match dtype {
        DataType::Int64 => {
            let mut builder =
                ListPrimitiveChunkedBuilder::<Int64Type>::new(name, len, len * 5, DataType::Int64);
            for sub_seq in seq.iter()? {
                let sub_seq = sub_seq?;
                let len = sub_seq.len()?;

                // safety: we know the iterators len
                let iter = unsafe {
                    sub_seq
                        .iter()?
                        .map(|v| {
                            let v = v.unwrap();
                            if v.is_none() {
                                None
                            } else {
                                Some(v.extract::<i64>().unwrap())
                            }
                        })
                        .trust_my_length(len)
                };
                builder.append_iter(iter)
            }
            builder.finish().into_series()
        }
        DataType::Float64 => {
            let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
                name,
                len,
                len * 5,
                DataType::Float64,
            );
            for sub_seq in seq.iter()? {
                let sub_seq = sub_seq?;
                let len = sub_seq.len()?;
                // safety: we know the iterators len
                let iter = unsafe {
                    sub_seq
                        .iter()?
                        .map(|v| {
                            let v = v.unwrap();
                            if v.is_none() {
                                None
                            } else {
                                Some(v.extract::<f64>().unwrap())
                            }
                        })
                        .trust_my_length(len)
                };
                builder.append_iter(iter)
            }
            builder.finish().into_series()
        }
        DataType::Boolean => {
            let mut builder = ListBooleanChunkedBuilder::new(name, len, len * 5);
            for sub_seq in seq.iter()? {
                let sub_seq = sub_seq?;
                let len = sub_seq.len()?;
                // safety: we know the iterators len
                let iter = unsafe {
                    sub_seq
                        .iter()?
                        .map(|v| {
                            let v = v.unwrap();
                            if v.is_none() {
                                None
                            } else {
                                Some(v.extract::<bool>().unwrap())
                            }
                        })
                        .trust_my_length(len)
                };
                builder.append_iter(iter)
            }
            builder.finish().into_series()
        }
        DataType::Utf8 => {
            let mut builder = ListUtf8ChunkedBuilder::new(name, len, len * 5);
            for sub_seq in seq.iter()? {
                let sub_seq = sub_seq?;
                let len = sub_seq.len()?;
                // safety: we know the iterators len
                let iter = unsafe {
                    sub_seq
                        .iter()?
                        .map(|v| {
                            let v = v.unwrap();
                            if v.is_none() {
                                None
                            } else {
                                Some(v.extract::<&str>().unwrap())
                            }
                        })
                        .trust_my_length(len)
                };
                builder.append_trusted_len_iter(iter)
            }
            builder.finish().into_series()
        }
        dt => {
            panic!("cannot create list array from {dt:?}");
        }
    };

    Ok(s)
}
