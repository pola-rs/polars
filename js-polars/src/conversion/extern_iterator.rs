pub struct ExternIterator<'a, T> {
    pub count: usize,
    pub len: usize,
    pub array: &'a T,
}

#[macro_export]
macro_rules! extern_iterator {
    ( $x:ident, $y:ty, $z:ty) => {
        paste::paste! {
        use crate::conversion::extern_struct::IntoRustStruct;
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(method, getter = length)]
            fn [<$x _length>](this: &$x) -> usize;
            #[wasm_bindgen(method, indexing_getter)]
            fn [<$x _get>](this: &$x, prop: usize) -> wasm_bindgen::JsValue;
        }

        impl<'a> Iterator for crate::conversion::extern_iterator::ExternIterator<'a, $x> {
            type Item = $z;
            fn next(&mut self) -> Option<Self::Item> {
                if self.count < self.len {
                    self.count += 1;
                    Some($y::from(self.array.[<$x _get>]( self.count)).into_rust())
                } else {
                    None
                }
            }
        }

        impl<'a> IntoIterator for &'a $x {
            type Item = $z;
            type IntoIter = crate::conversion::extern_iterator::ExternIterator<'a, $x>;
            fn into_iter(self) -> Self::IntoIter {
                crate::conversion::extern_iterator::ExternIterator {
                    count: 0,
                    len: self.[<$x _length>](),
                    array: self,
                }
            }
        }
        }
    };
}
