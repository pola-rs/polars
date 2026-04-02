use polars_utils::pl_str::PlSmallStr;

pub const CSE_REPLACED: &str = "__POLARS_CSER_";
pub const POLARS_TMP_PREFIX: &str = "_POLARS_";
pub const POLARS_PLACEHOLDER: &str = "_POLARS_<>";
pub const POLARS_ELEMENT: &str = "__PL_ELEMENT";
pub const POLARS_STRUCTFIELDS: &str = "__PL_STRUCTFIELDS";
pub const LEN: &str = "len";

const LITERAL_NAME: PlSmallStr = PlSmallStr::from_static("literal");
const LEN_NAME: PlSmallStr = PlSmallStr::from_static(LEN);
const PL_ELEMENT_NAME: PlSmallStr = PlSmallStr::from_static(POLARS_ELEMENT);
const PL_STRUCTFIELDS_NAME: PlSmallStr = PlSmallStr::from_static(POLARS_STRUCTFIELDS);

pub fn get_literal_name() -> PlSmallStr {
    LITERAL_NAME.clone()
}

pub(crate) fn get_len_name() -> PlSmallStr {
    LEN_NAME.clone()
}

pub fn get_pl_element_name() -> PlSmallStr {
    PL_ELEMENT_NAME.clone()
}

pub fn get_pl_structfields_name() -> PlSmallStr {
    PL_STRUCTFIELDS_NAME.clone()
}
