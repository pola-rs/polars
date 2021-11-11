use crate::conversion::get_pyseq;
use crate::utils::str_to_polarstype;
use polars::prelude::*;
use polars_core::utils::CustomIterTools;
use pyo3::{PyAny, PyResult};

pub fn py_seq_to_list(name: &str, seq: &PyAny, dtype: &PyAny) -> PyResult<Series> {
    let str_repr = dtype.str().unwrap().to_str().unwrap();
    let dtype = str_to_polarstype(str_repr);

    let (seq, len) = get_pyseq(seq)?;
    let s = match dtype {
        DataType::Int64 => {
            let mut builder =
                ListPrimitiveChunkedBuilder::<i64>::new(name, len, len * 5, DataType::Int64);
            for sub_seq in seq.iter()? {
                let sub_seq = sub_seq?;
                let (sub_seq, len) = get_pyseq(sub_seq)?;
                let iter = sub_seq
                    .iter()?
                    .map(|v| {
                        let v = v.unwrap();
                        if v.is_none() {
                            None
                        } else {
                            Some(v.extract::<i64>().unwrap())
                        }
                    })
                    .trust_my_length(len);
                builder.append_iter(iter)
            }
            builder.finish().into_series()
        }
        DataType::Float64 => {
            let mut builder =
                ListPrimitiveChunkedBuilder::<f64>::new(name, len, len * 5, DataType::Float64);
            for sub_seq in seq.iter()? {
                let sub_seq = sub_seq?;
                let (sub_seq, len) = get_pyseq(sub_seq)?;
                let iter = sub_seq
                    .iter()?
                    .map(|v| {
                        let v = v.unwrap();
                        if v.is_none() {
                            None
                        } else {
                            Some(v.extract::<f64>().unwrap())
                        }
                    })
                    .trust_my_length(len);
                builder.append_iter(iter)
            }
            builder.finish().into_series()
        }
        DataType::Boolean => {
            let mut builder = ListBooleanChunkedBuilder::new(name, len, len * 5);
            for sub_seq in seq.iter()? {
                let sub_seq = sub_seq?;
                let (sub_seq, len) = get_pyseq(sub_seq)?;
                let iter = sub_seq
                    .iter()?
                    .map(|v| {
                        let v = v.unwrap();
                        if v.is_none() {
                            None
                        } else {
                            Some(v.extract::<bool>().unwrap())
                        }
                    })
                    .trust_my_length(len);
                builder.append_iter(iter)
            }
            builder.finish().into_series()
        }
        DataType::Utf8 => {
            let mut builder = ListUtf8ChunkedBuilder::new(name, len, len * 5);
            for sub_seq in seq.iter()? {
                let sub_seq = sub_seq?;
                let (sub_seq, len) = get_pyseq(sub_seq)?;
                let iter = sub_seq
                    .iter()?
                    .map(|v| {
                        let v = v.unwrap();
                        if v.is_none() {
                            None
                        } else {
                            Some(v.extract::<&str>().unwrap())
                        }
                    })
                    .trust_my_length(len);
                builder.append_iter(iter)
            }
            builder.finish().into_series()
        }
        dt => {
            panic!("cannot create list array from {:?}", dt);
        }
    };

    Ok(s)
}
