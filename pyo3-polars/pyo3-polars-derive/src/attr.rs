use std::fmt::Debug;

use proc_macro2::Ident;
use syn::parse::{Parse, ParseStream};
use syn::Token;

use crate::keywords;

#[derive(Clone, Debug)]
pub struct KeyWordAttribute<K, V> {
    #[allow(dead_code)]
    pub kw: K,
    pub value: V,
}

impl<K: Parse, V: Parse> Parse for KeyWordAttribute<K, V> {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let kw = input.parse()?;
        let _: Token![=] = input.parse()?;
        let value = input.parse()?;
        Ok(KeyWordAttribute { kw, value })
    }
}

pub type OutputAttribute = KeyWordAttribute<keywords::output_type, Ident>;
pub type OutputFuncAttribute = KeyWordAttribute<keywords::output_type_func, Ident>;
pub type OutputFuncAttributeWithKwargs =
    KeyWordAttribute<keywords::output_type_func_with_kwargs, Ident>;

#[derive(Default, Debug)]
pub struct ExprsFunctionOptions {
    pub output_dtype: Option<Ident>,
    pub output_type_fn: Option<Ident>,
    pub output_type_fn_kwargs: Option<Ident>,
}

impl Parse for ExprsFunctionOptions {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut options = ExprsFunctionOptions::default();

        while !input.is_empty() {
            let lookahead = input.lookahead1();

            if lookahead.peek(keywords::output_type) {
                let attr = input.parse::<OutputAttribute>()?;
                options.output_dtype = Some(attr.value)
            } else if lookahead.peek(keywords::output_type_func) {
                let attr = input.parse::<OutputFuncAttribute>()?;
                options.output_type_fn = Some(attr.value)
            } else if lookahead.peek(keywords::output_type_func_with_kwargs) {
                let attr = input.parse::<OutputFuncAttributeWithKwargs>()?;
                options.output_type_fn_kwargs = Some(attr.value)
            } else {
                panic!("didn't recognize attribute")
            }
        }
        Ok(options)
    }
}
