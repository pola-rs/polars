mod attr;
mod keywords;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, FnArg};

fn quote_get_kwargs() -> proc_macro2::TokenStream {
    quote!(
    let kwargs = std::slice::from_raw_parts(kwargs_ptr, kwargs_len);

    let kwargs = match pyo3_polars::derive::_parse_kwargs(kwargs)  {
        Ok(value) => value,
        Err(err) => {
            let err = polars_error::polars_err!(InvalidOperation: "could not parse kwargs: '{}'\n\nCheck: registration of kwargs in the plugin.", err);
            pyo3_polars::derive::_update_last_error(err);
            return;
        }
    };

    )
}

fn quote_call_kwargs(ast: &syn::ItemFn, fn_name: &syn::Ident) -> proc_macro2::TokenStream {
    let kwargs = quote_get_kwargs();
    quote!(
            // parse the kwargs and assign to `let kwargs`
            #kwargs

            // define the function
            #ast

            // call the function
        let result: polars_error::PolarsResult<polars_core::prelude::Series> = #fn_name(&inputs, kwargs);

    )
}

fn quote_call_context(ast: &syn::ItemFn, fn_name: &syn::Ident) -> proc_macro2::TokenStream {
    quote!(
            let context = *context;

            // define the function
            #ast

            // call the function
        let result: polars_error::PolarsResult<polars_core::prelude::Series> = #fn_name(&inputs, context);
    )
}

fn quote_call_context_kwargs(ast: &syn::ItemFn, fn_name: &syn::Ident) -> proc_macro2::TokenStream {
    quote!(
            let context = *context;

            let kwargs = std::slice::from_raw_parts(kwargs_ptr, kwargs_len);

            let kwargs = match pyo3_polars::derive::_parse_kwargs(kwargs)  {
                    Ok(value) => value,
                    Err(err) => {
                        pyo3_polars::derive::_update_last_error(err);
                        return;
                    }
            };

            // define the function
            #ast

            // call the function
        let result: polars_error::PolarsResult<polars_core::prelude::Series> = #fn_name(&inputs, context, kwargs);
    )
}

fn quote_call_no_kwargs(ast: &syn::ItemFn, fn_name: &syn::Ident) -> proc_macro2::TokenStream {
    quote!(
            // define the function
            #ast
            // call the function
            let result: polars_error::PolarsResult<polars_core::prelude::Series> = #fn_name(&inputs);
    )
}

fn quote_process_results() -> proc_macro2::TokenStream {
    quote!(match result {
        Ok(out) => {
            // Update return value.
            *return_value = polars_ffi::version_0::export_series(&out);
        },
        Err(err) => {
            // Set latest error, but leave return value in empty state.
            pyo3_polars::derive::_update_last_error(err);
        },
    })
}

fn create_expression_function(ast: syn::ItemFn) -> proc_macro2::TokenStream {
    // count how often the user define a kwargs argument.
    let args = ast
        .sig
        .inputs
        .iter()
        .skip(1)
        .map(|fn_arg| {
            if let FnArg::Typed(pat) = fn_arg {
                if let syn::Pat::Ident(pat) = pat.pat.as_ref() {
                    pat.ident.to_string()
                } else {
                    panic!("expected an argument")
                }
            } else {
                panic!("expected a type argument")
            }
        })
        .collect::<Vec<_>>();

    let fn_name = &ast.sig.ident;

    // Get the tokenstream of the call logic.
    let quote_call = match args.len() {
        0 => quote_call_no_kwargs(&ast, fn_name),
        1 => match args[0].as_str() {
            "kwargs" => quote_call_kwargs(&ast, fn_name),
            "context" => quote_call_context(&ast, fn_name),
            a => panic!("didn't expect argument {a}"),
        },
        2 => match (args[0].as_str(), args[1].as_str()) {
            ("context", "kwargs") => quote_call_context_kwargs(&ast, fn_name),
            ("kwargs", "context") => panic!("'kwargs', 'context' order should be reversed"),
            (a, b) => panic!("didn't expect arguments {a}, {b}"),
        },
        _ => panic!("didn't expect so many arguments"),
    };

    let quote_process_result = quote_process_results();
    let fn_name = get_expression_function_name(fn_name);

    quote!(
        use ::pyo3_polars::export::*;

        // create the outer public function
        #[no_mangle]
        pub unsafe extern "C" fn #fn_name (
            e: *mut polars_ffi::version_0::SeriesExport,
            input_len: usize,
            kwargs_ptr: *const u8,
            kwargs_len: usize,
            return_value: *mut polars_ffi::version_0::SeriesExport,
            context: *mut polars_ffi::version_0::CallerContext
        )  {
            let panic_result = std::panic::catch_unwind(move || {
                let inputs = polars_ffi::version_0::import_series_buffer(e, input_len).unwrap();

                #quote_call

                #quote_process_result
            });

            if panic_result.is_err() {
                // Set latest to panic;
                ::pyo3_polars::derive::_set_panic();
            }
        }
    )
}

fn get_field_function_name(fn_name: &syn::Ident) -> syn::Ident {
    syn::Ident::new(&format!("_polars_plugin_field_{fn_name}"), fn_name.span())
}

fn get_expression_function_name(fn_name: &syn::Ident) -> syn::Ident {
    syn::Ident::new(&format!("_polars_plugin_{fn_name}"), fn_name.span())
}

fn quote_get_inputs() -> proc_macro2::TokenStream {
    quote!(
             let inputs = std::slice::from_raw_parts(field, len);
             let inputs = inputs.iter().map(|field| {
                 let field = polars_arrow::ffi::import_field_from_c(field).unwrap();
                 let out = polars_core::prelude::Field::from(&field);
                 out
             }).collect::<Vec<_>>();
    )
}

fn create_field_function(
    fn_name: &syn::Ident,
    dtype_fn_name: &syn::Ident,
    kwargs: bool,
) -> proc_macro2::TokenStream {
    let map_field_name = get_field_function_name(fn_name);
    let inputs = quote_get_inputs();

    let call_fn = if kwargs {
        let kwargs = quote_get_kwargs();
        quote! (
            #kwargs
            let result = #dtype_fn_name(&inputs, kwargs);
        )
    } else {
        quote!(
            let result = #dtype_fn_name(&inputs);
        )
    };

    quote! (
        #[no_mangle]
        pub unsafe extern "C" fn #map_field_name(
            field: *mut polars_arrow::ffi::ArrowSchema,
            len: usize,
            return_value: *mut polars_arrow::ffi::ArrowSchema,
            kwargs_ptr: *const u8,
            kwargs_len: usize,
        ) {
            let panic_result = std::panic::catch_unwind(move || {
                #inputs;

                #call_fn;

                match result {
                    Ok(out) => {
                        let out = polars_arrow::ffi::export_field_to_c(&out.to_arrow(polars_core::datatypes::CompatLevel::newest()));
                        *return_value = out;
                    },
                    Err(err) => {
                        // Set latest error, but leave return value in empty state.
                        pyo3_polars::derive::_update_last_error(err);
                    }
                }
            });

            if panic_result.is_err() {
                // Set latest to panic;
                pyo3_polars::derive::_set_panic();
            }
        }
    )
}

fn create_field_function_from_with_dtype(
    fn_name: &syn::Ident,
    dtype: syn::Ident,
) -> proc_macro2::TokenStream {
    let map_field_name = get_field_function_name(fn_name);
    let inputs = quote_get_inputs();

    quote! (
        #[no_mangle]
        pub unsafe extern "C" fn #map_field_name(
            field: *mut polars_arrow::ffi::ArrowSchema,
            len: usize,
            return_value: *mut polars_arrow::ffi::ArrowSchema
        ) {
            #inputs

            let mapper = polars_plan::prelude::FieldsMapper::new(&inputs);
            let dtype = polars_core::datatypes::DataType::#dtype;
            let out = mapper.with_dtype(dtype).unwrap();
            let out = polars_arrow::ffi::export_field_to_c(&out.to_arrow(polars_core::datatypes::CompatLevel::newest()));
            *return_value = out;
        }
    )
}

#[proc_macro_attribute]
pub fn polars_expr(attr: TokenStream, input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as syn::ItemFn);

    let options = parse_macro_input!(attr as attr::ExprsFunctionOptions);
    let expanded_field_fn = if let Some(fn_name) = options.output_type_fn {
        create_field_function(&ast.sig.ident, &fn_name, false)
    } else if let Some(fn_name) = options.output_type_fn_kwargs {
        create_field_function(&ast.sig.ident, &fn_name, true)
    } else if let Some(dtype) = options.output_dtype {
        create_field_function_from_with_dtype(&ast.sig.ident, dtype)
    } else {
        panic!("didn't understand polars_expr attribute")
    };

    let expanded_expr = create_expression_function(ast);
    let expanded = quote!(
        #expanded_field_fn

        #expanded_expr
    );
    TokenStream::from(expanded)
}

fn create_rewrite_function(ast: &syn::ItemFn) -> proc_macro2::TokenStream {
    let args = ast
        .sig
        .inputs
        .iter()
        .skip(1)
        .map(|fn_arg| match fn_arg {
            FnArg::Typed(pat) => match pat.pat.as_ref() {
                syn::Pat::Ident(pat) => pat.ident.to_string(),
                _ => panic!("expected an argument"),
            },
            _ => panic!("expected a type argument"),
        })
        .collect::<Vec<_>>();
    let args = args.iter().map(String::as_str).collect::<Vec<_>>();

    let fn_name = &ast.sig.ident;
    let call = match args.as_slice() {
        [] => quote!(#fn_name(&inputs)),
        ["kwargs"] => quote!(#fn_name(&inputs, kwargs)),
        ["context"] => quote!(#fn_name(&inputs, context)),
        ["context", "kwargs"] => quote!(#fn_name(&inputs, context, kwargs)),
        ["kwargs", "context"] => panic!("'kwargs', 'context' order should be reversed"),
        a => panic!("didn't expect arguments {a:?}"),
    };
    let get_context = if args.contains(&"context") {
        quote!(
            let context = ::pyo3_polars::rewrite::RewriteContext::_new(
                ::pyo3_polars::rewrite::_import_lib(lib_ptr, lib_len),
            );
        )
    } else {
        quote!()
    };
    let get_kwargs = if args.contains(&"kwargs") {
        quote!(
            let kwargs = std::slice::from_raw_parts(kwargs_ptr, kwargs_len);
            let kwargs = match ::pyo3_polars::derive::_parse_kwargs(kwargs) {
                Ok(value) => value,
                Err(err) => {
                    let err = ::pyo3_polars::export::polars_error::polars_err!(InvalidOperation: "could not parse kwargs: '{}'\n\nCheck: registration of kwargs in the plugin.", err);
                    ::pyo3_polars::derive::_update_last_error(err);
                    return;
                },
            };
        )
    } else {
        quote!()
    };

    let extern_fn_name =
        syn::Ident::new(&format!("_polars_plugin_rewrite_{fn_name}"), fn_name.span());
    quote!(
        #[no_mangle]
        pub unsafe extern "C" fn #extern_fn_name(
            fields: *const ::pyo3_polars::export::polars_arrow::ffi::ArrowSchema,
            n_fields: usize,
            kwargs_ptr: *const u8,
            kwargs_len: usize,
            lib_ptr: *const u8,
            lib_len: usize,
            return_value: *mut ::pyo3_polars::export::polars_ffi::dsl_rewrite::BytesExport,
        ) {
            let panic_result = std::panic::catch_unwind(move || {
                let inputs = ::pyo3_polars::rewrite::_import_fields(fields, n_fields);
                #get_context
                #get_kwargs
                ::pyo3_polars::rewrite::_export_rewrite(#call, return_value);
            });

            if panic_result.is_err() {
                // Set latest to panic;
                ::pyo3_polars::derive::_set_panic();
            }
        }
    )
}

/// Export function which can rewrite an expression at planning time.
///
/// The function takes (in this order)
/// - the input fields
/// - (opt) a `context: RewriteContext`
/// - (opt) `kwargs`
/// The function itself stays available, the
/// macro adds the `_polars_plugin_rewrite_{name}` symbol that Polars calls.
#[proc_macro_attribute]
pub fn polars_rewrite(attr: TokenStream, input: TokenStream) -> TokenStream {
    assert!(attr.is_empty(), "polars_rewrite takes no arguments");
    let ast = parse_macro_input!(input as syn::ItemFn);
    let expanded_rewrite = create_rewrite_function(&ast);
    TokenStream::from(quote!(
        #ast

        #expanded_rewrite
    ))
}
