use arrow::array::{
    make_array, Array, ArrayData, ArrayRef, Capacities, MutableArrayData, StringOffsetSizeTrait,
};
use arrow::datatypes::DataType;

fn compute_str_values_length<Offset: StringOffsetSizeTrait>(data: &ArrayData) -> usize {
    // get the length of the value buffer
    let buf_len = data.buffers()[1].len();
    // find the offset of the buffer
    // this returns a slice of offsets, starting from the offset of the array
    // so we can take the first value
    unsafe {
        let ptr = data.buffers()[0].as_ptr().add(data.offset()) as *const Offset;
        let offset = *ptr;
        buf_len - offset.to_usize().unwrap()
    }
}

pub fn shrink_to_fit(array: &dyn Array) -> ArrayRef {
    let length = array.len();
    let capacity = length;
    let data = array.data();

    let mut mutable = match array.data().data_type() {
        DataType::Utf8 => {
            let str_values_size = compute_str_values_length::<i32>(data);
            MutableArrayData::with_capacities(
                vec![data],
                false,
                Capacities::Binary(capacity, Some(str_values_size)),
            )
        }
        DataType::LargeUtf8 => {
            let str_values_size = compute_str_values_length::<i64>(data);
            MutableArrayData::with_capacities(
                vec![data],
                false,
                Capacities::Binary(capacity, Some(str_values_size)),
            )
        }
        _ => MutableArrayData::new(vec![data], false, capacity),
    };

    mutable.extend(0, 0, length);

    make_array(mutable.freeze())
}
