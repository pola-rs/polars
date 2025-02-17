//! APIs to write to Avro format.
use avro_schema::file::Block;

mod schema;
pub use schema::to_record;
mod serialize;
pub use serialize::{can_serialize, new_serializer, BoxSerializer};

/// consumes a set of [`BoxSerializer`] into an [`Block`].
/// # Panics
/// Panics iff the number of items in any of the serializers is not equal to the number of rows
/// declared in the `block`.
pub fn serialize(serializers: &mut [BoxSerializer], block: &mut Block) {
    let Block {
        data,
        number_of_rows,
    } = block;

    data.clear(); // restart it

    // _the_ transpose (columns -> rows)
    for _ in 0..*number_of_rows {
        for serializer in &mut *serializers {
            let item_data = serializer.next().unwrap();
            data.extend(item_data);
        }
    }
}
