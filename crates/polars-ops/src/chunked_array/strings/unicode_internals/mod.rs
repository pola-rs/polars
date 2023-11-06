mod unicode_data;

// For use in alloc, not re-exported in std.
pub(super) use unicode_data::case_ignorable::lookup as Case_Ignorable;
pub(super) use unicode_data::cased::lookup as Cased;
