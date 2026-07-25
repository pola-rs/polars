//! Deprecated compatibility wrapper for the Polars expression-plugin macro.

use proc_macro::TokenStream;
use quote::quote;

#[deprecated(
    since = "0.21.0",
    note = "use `polars_plugin::polars_expr` or `pyo3_polars::derive::polars_expr`"
)]
#[proc_macro_attribute]
pub fn polars_expr(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = proc_macro2::TokenStream::from(attr);
    let item = proc_macro2::TokenStream::from(item);
    quote!(
        #[::pyo3_polars::derive::polars_expr(#attr)]
        #item
    )
    .into()
}
