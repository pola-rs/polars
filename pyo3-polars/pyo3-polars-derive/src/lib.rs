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

fn quote_call_kwargs(fn_name: &syn::Ident) -> proc_macro2::TokenStream {
    let kwargs = quote_get_kwargs();
    quote!(
            // parse the kwargs and assign to `let kwargs`
            #kwargs

            // call the function
        let result = #fn_name(&inputs, kwargs);

    )
}

fn quote_call_context(fn_name: &syn::Ident) -> proc_macro2::TokenStream {
    quote!(
            let context = *context;

            // call the function
        let result = #fn_name(&inputs, context);
    )
}

fn quote_call_context_kwargs(fn_name: &syn::Ident) -> proc_macro2::TokenStream {
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

            // call the function
        let result = #fn_name(&inputs, context, kwargs);
    )
}

fn quote_call_no_kwargs(fn_name: &syn::Ident) -> proc_macro2::TokenStream {
    quote!(
            // call the function
            let result = #fn_name(&inputs);
    )
}

fn quote_process_results(export_output: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    quote!(match result {
        Ok(out) => {
            // Update return value.
            *return_value = #export_output;
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
    let takes_columns = takes_columns(&ast);
    let output_type = if takes_columns {
        quote!(polars_core::prelude::Column)
    } else {
        quote!(polars_core::prelude::Series)
    };

    // Get the tokenstream of the call logic.
    let quote_call = match args.len() {
        0 => quote_call_no_kwargs(fn_name),
        1 => match args[0].as_str() {
            "kwargs" => quote_call_kwargs(fn_name),
            "context" => quote_call_context(fn_name),
            a => panic!("didn't expect argument {a}"),
        },
        2 => match (args[0].as_str(), args[1].as_str()) {
            ("context", "kwargs") => quote_call_context_kwargs(fn_name),
            ("kwargs", "context") => panic!("'kwargs', 'context' order should be reversed"),
            (a, b) => panic!("didn't expect arguments {a}, {b}"),
        },
        _ => panic!("didn't expect so many arguments"),
    };
    // Functions taking `&[Column]` may return either a `Series` or a `Column`.
    let quote_call = quote!(
        #quote_call
        let result: polars_error::PolarsResult<#output_type> = result.map(<#output_type>::from);
    );

    let import_series = quote!(polars_ffi::version_0::import_series_buffer(e, input_len).unwrap());
    let series_function = if takes_columns {
        // Called by Polars versions which don't know about column functions.
        quote_expression_function(
            &get_expression_function_name(fn_name),
            &quote!(polars_ffi::version_0::SeriesExport),
            &quote!(#import_series.into_iter().map(polars_core::prelude::Column::from).collect::<Vec<_>>()),
            &quote_call,
            &quote!(polars_ffi::version_0::export_series(
                out.as_materialized_series()
            )),
        )
    } else {
        quote_expression_function(
            &get_expression_function_name(fn_name),
            &quote!(polars_ffi::version_0::SeriesExport),
            &import_series,
            &quote_call,
            &quote!(polars_ffi::version_0::export_series(&out)),
        )
    };
    let column_function = takes_columns.then(|| {
        quote_expression_function(
            &get_column_function_name(fn_name),
            &quote!(polars_ffi::version_0::ColumnExport),
            &quote!(polars_ffi::version_0::import_column_buffer(e, input_len).unwrap()),
            &quote_call,
            &quote!(polars_ffi::version_0::export_column(&out)),
        )
    });

    quote!(
        use ::pyo3_polars::export::*;

        #ast

        #series_function

        #column_function
    )
}

/// Whether the inputs are taken as `&[Column]` instead of `&[Series]`.
fn takes_columns(ast: &syn::ItemFn) -> bool {
    let Some(FnArg::Typed(inputs)) = ast.sig.inputs.first() else {
        return false;
    };
    let syn::Type::Reference(reference) = inputs.ty.as_ref() else {
        return false;
    };
    let syn::Type::Slice(slice) = reference.elem.as_ref() else {
        return false;
    };
    let syn::Type::Path(path) = slice.elem.as_ref() else {
        return false;
    };
    path.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "Column")
}

fn quote_expression_function(
    name: &syn::Ident,
    export_type: &proc_macro2::TokenStream,
    import_inputs: &proc_macro2::TokenStream,
    quote_call: &proc_macro2::TokenStream,
    export_output: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let quote_process_result = quote_process_results(export_output);
    quote!(
        #[no_mangle]
        pub unsafe extern "C" fn #name (
            e: *mut #export_type,
            input_len: usize,
            kwargs_ptr: *const u8,
            kwargs_len: usize,
            return_value: *mut #export_type,
            context: *mut polars_ffi::version_0::CallerContext
        )  {
            let panic_result = std::panic::catch_unwind(move || {
                let inputs = #import_inputs;

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

fn get_column_function_name(fn_name: &syn::Ident) -> syn::Ident {
    syn::Ident::new(&format!("_polars_column_plugin_{fn_name}"), fn_name.span())
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
