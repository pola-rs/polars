use crate::array::binview::BinaryViewArrayGeneric;
use crate::bitmap::MutableBitmap;

#[derive(Debug, Clone)]
pub struct MutableBinaryViewArray<const IS_UTF: bool> {
    views: Vec<u128>,
    buffer: Vec<u8>,
    validity: Option<MutableBitmap>,
}

// impl<const IS_UTF8: bool> From<MutableBinaryViewArray<IS_UTF8>> for BinaryViewArray<IS_UTF8> {
//     fn from(value: MutableBinaryViewArray<IS_UTF8>) -> Self {
//         if IS_UTF8 {
//
//         }
//         BinaryViewArray::new_unchecked()
//         todo!()
//     }
// }
