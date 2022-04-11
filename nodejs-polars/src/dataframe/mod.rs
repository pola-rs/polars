pub mod frame;
pub mod io;

use frame as df;
pub use frame::JsDataFrame;
pub use frame::*;

use crate::prelude::JsResult;
use napi::JsObject;

impl JsDataFrame {
    pub fn to_object(env: &napi::Env) -> JsResult<JsObject> {
        let mut df_obj = env.create_object()?;

        df_obj.define_properties(&[
            napi::Property::new(env, "lazy")?.with_method(df::lazy),
            napi::Property::new(env, "add")?.with_method(df::add),
            napi::Property::new(env, "as_str")?.with_method(df::as_str),
            napi::Property::new(env, "clone")?.with_method(df::clone),
            napi::Property::new(env, "column")?.with_method(df::column),
            napi::Property::new(env, "columns")?.with_method(df::columns),
            napi::Property::new(env, "div")?.with_method(df::div),
            napi::Property::new(env, "drop_in_place")?.with_method(df::drop_in_place),
            napi::Property::new(env, "drop_nulls")?.with_method(df::drop_nulls),
            napi::Property::new(env, "unique")?.with_method(df::unique),
            napi::Property::new(env, "drop")?.with_method(df::drop),
            napi::Property::new(env, "dtypes")?.with_method(df::dtypes),
            napi::Property::new(env, "extend")?.with_method(df::extend),
            napi::Property::new(env, "fill_null")?.with_method(df::fill_null),
            napi::Property::new(env, "filter")?.with_method(df::filter),
            napi::Property::new(env, "find_idx_by_name")?.with_method(df::find_idx_by_name),
            napi::Property::new(env, "frame_equal")?.with_method(df::frame_equal),
            napi::Property::new(env, "get_columns")?.with_method(df::get_columns),
            napi::Property::new(env, "groupby")?.with_method(df::groupby),
            napi::Property::new(env, "hash_rows")?.with_method(df::hash_rows),
            napi::Property::new(env, "head")?.with_method(df::head),
            napi::Property::new(env, "height")?.with_method(df::height),
            napi::Property::new(env, "hstack")?.with_method(df::hstack),
            napi::Property::new(env, "insert_at_idx")?.with_method(df::insert_at_idx),
            napi::Property::new(env, "is_duplicated")?.with_method(df::is_duplicated),
            napi::Property::new(env, "is_unique")?.with_method(df::is_unique),
            napi::Property::new(env, "join")?.with_method(df::join),
            napi::Property::new(env, "max")?.with_method(df::max),
            napi::Property::new(env, "mean")?.with_method(df::mean),
            napi::Property::new(env, "median")?.with_method(df::median),
            napi::Property::new(env, "melt")?.with_method(df::melt),
            napi::Property::new(env, "min")?.with_method(df::min),
            napi::Property::new(env, "mul")?.with_method(df::mul),
            napi::Property::new(env, "n_chunks")?.with_method(df::n_chunks),
            napi::Property::new(env, "null_count")?.with_method(df::null_count),
            napi::Property::new(env, "hmax")?.with_method(df::hmax),
            napi::Property::new(env, "hmean")?.with_method(df::hmean),
            napi::Property::new(env, "hmin")?.with_method(df::hmin),
            napi::Property::new(env, "hsum")?.with_method(df::hsum),
            napi::Property::new(env, "quantile")?.with_method(df::quantile),
            napi::Property::new(env, "rechunk")?.with_method(df::rechunk),
            napi::Property::new(env, "rem")?.with_method(df::rem),
            napi::Property::new(env, "rename")?.with_method(df::rename),
            napi::Property::new(env, "replace_at_idx")?.with_method(df::replace_at_idx),
            napi::Property::new(env, "replace")?.with_method(df::replace),
            napi::Property::new(env, "sample_frac")?.with_method(df::sample_frac),
            napi::Property::new(env, "sample_n")?.with_method(df::sample_n),
            napi::Property::new(env, "schema")?.with_method(df::schema),
            napi::Property::new(env, "select_at_idx")?.with_method(df::select_at_idx),
            napi::Property::new(env, "select")?.with_method(df::select),
            napi::Property::new(env, "set_column_names")?.with_method(df::set_column_names),
            napi::Property::new(env, "shape")?.with_method(df::shape),
            napi::Property::new(env, "shift")?.with_method(df::shift),
            napi::Property::new(env, "shrink_to_fit")?.with_method(df::shrink_to_fit),
            napi::Property::new(env, "slice")?.with_method(df::slice),
            napi::Property::new(env, "sort_in_place")?.with_method(df::sort_in_place),
            napi::Property::new(env, "sort")?.with_method(df::sort),
            napi::Property::new(env, "std")?.with_method(df::std),
            napi::Property::new(env, "sub")?.with_method(df::sub),
            napi::Property::new(env, "sum")?.with_method(df::sum),
            napi::Property::new(env, "tail")?.with_method(df::tail),
            napi::Property::new(env, "take_with_series")?.with_method(df::take_with_series),
            napi::Property::new(env, "take")?.with_method(df::take),
            napi::Property::new(env, "transpose")?.with_method(df::transpose),
            napi::Property::new(env, "var")?.with_method(df::var),
            napi::Property::new(env, "vstack")?.with_method(df::vstack),
            napi::Property::new(env, "width")?.with_method(df::width),
            napi::Property::new(env, "with_column")?.with_method(df::with_column),
            napi::Property::new(env, "with_row_count")?.with_method(df::with_row_count),
            // IO
            napi::Property::new(env, "to_bincode")?.with_method(io::to_bincode),
            napi::Property::new(env, "from_bincode")?.with_method(io::from_bincode),
            napi::Property::new(env, "to_js")?.with_method(io::to_js),
            // row
            napi::Property::new(env, "to_row")?.with_method(io::to_row),
            napi::Property::new(env, "to_row_object")?.with_method(io::to_row_object),
            // rows
            napi::Property::new(env, "to_rows")?.with_method(io::to_rows),
            napi::Property::new(env, "read_rows")?.with_method(io::read_rows),
            napi::Property::new(env, "read_array_rows")?.with_method(io::read_array_rows),
            napi::Property::new(env, "to_row_objects")?.with_method(io::to_row_objects),
            napi::Property::new(env, "read_columns")?.with_method(io::read_columns),
            //csv
            napi::Property::new(env, "readCSVBuffer")?.with_method(io::read_csv_buffer),
            napi::Property::new(env, "readCSVPath")?.with_method(io::read_csv_path),
            napi::Property::new(env, "write_csv_path")?.with_method(io::write_csv_path),
            napi::Property::new(env, "write_csv_stream")?.with_method(io::write_csv_stream),
            // json
            napi::Property::new(env, "to_json")?.with_method(io::to_json),
            napi::Property::new(env, "readJSONBuffer")?.with_method(io::read_json_buffer),
            napi::Property::new(env, "readJSONPath")?.with_method(io::read_json_path),
            napi::Property::new(env, "write_json_path")?.with_method(io::write_json_path),
            napi::Property::new(env, "write_json_stream")?.with_method(io::write_json_stream),
            // parquet
            napi::Property::new(env, "readParquetPath")?.with_method(io::read_parquet_path),
            napi::Property::new(env, "readParquetBuffer")?.with_method(io::read_parquet_buffer),
            napi::Property::new(env, "write_parquet_path")?.with_method(io::write_parquet_path),
            napi::Property::new(env, "write_parquet_stream")?.with_method(io::write_parquet_stream),
            // ipc
            napi::Property::new(env, "readIPCPath")?.with_method(io::read_ipc_path),
            napi::Property::new(env, "readIPCBuffer")?.with_method(io::read_ipc_buffer),
            napi::Property::new(env, "write_ipc_path")?.with_method(io::write_ipc_path),
            napi::Property::new(env, "write_ipc_stream")?.with_method(io::write_ipc_stream),

            // avro
            napi::Property::new(env, "readAvroPath")?.with_method(io::read_avro_path),
            napi::Property::new(env, "readAvroBuffer")?.with_method(io::read_avro_buffer),
            napi::Property::new(env, "write_avro_path")?.with_method(io::write_avro_path),
            napi::Property::new(env, "write_avro_stream")?.with_method(io::write_avro_stream),
        ])?;
        Ok(df_obj)
    }
}
