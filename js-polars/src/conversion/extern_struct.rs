use wasm_bindgen::convert::{FromWasmAbi, RefFromWasmAbi};

pub trait IntoRustStruct<T: FromWasmAbi<Abi = u32>> {
    fn into_rust(self) -> T;
}

pub trait RefRustStruct<'a, T: RefFromWasmAbi<Abi = u32>> {
    fn ref_rust(self) -> wasm_bindgen::__rt::Ref<'static, T>;
}

#[macro_export]
macro_rules! extern_struct {
    ( $x:ident, $y:ty) => {
        paste::paste! {
            use wasm_bindgen::convert::{FromWasmAbi, RefFromWasmAbi};
            #[wasm_bindgen]
            extern "C" {
                #[wasm_bindgen(method, getter = ptr)]
                fn [<$x _ptr>](this: &$x) -> f64;
            }

            impl<'a> crate::conversion::extern_struct::IntoRustStruct<$y> for $x {
                fn into_rust(self) -> $y {
                    unsafe { $y::from_abi(self.[<$x _ptr>]() as u32) }
                }
            }

            impl<'a> crate::conversion::extern_struct::RefRustStruct<'a, $y> for &'a $x {
                fn ref_rust(self) -> wasm_bindgen::__rt::Ref<'static, $y> {
                    unsafe { $y::ref_from_abi(self.[<$x _ptr>]() as u32) }
                }
            }
        }
    };
}
