//! Constant that help with creating error messages dependent on the host language.
#[cfg(feature = "python")]
pub static TRUE: &str = "True";
#[cfg(feature = "python")]
pub static FALSE: &str = "False";

#[cfg(not(feature = "python"))]
pub static TRUE: &str = "true";
#[cfg(not(feature = "python"))]
pub static FALSE: &str = "false";
