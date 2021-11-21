use crate::prelude::*;
use crate::chunked_array::object::extension::PolarsExtension;

pub(crate) fn drop_list(ca: &ListChunked) {
    let mut inner = ca.inner_dtype();
    let mut nested_count = 0;

    loop {
        match inner.inner_dtype() {
            Some(a) => {
                nested_count += 1;
                inner = a.clone()
            }
            None => {
                break;
            }
        }
    }

    if matches!(inner, DataType::Object(_)) {
        if nested_count != 0 {
            panic!("multiple nested objects not yet supported")
        }
        for arr in &ca.chunks {
            let dtype = arr.data_type();
            assert!(matches!(dtype, ArrowDataType::Extension(_, _, _)));

            // recreate the polars extension so that the content is dropped
            let arr = arr.as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap().clone();
            PolarsExtension::new(arr);
        }
    }
}
