use std::fmt::Debug;
use std::future::Future;
use std::hash::Hash;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
#[cfg(feature = "aws")]
pub use object_store::aws::AwsCredential;
#[cfg(feature = "azure")]
pub use object_store::azure::AzureCredential;
#[cfg(feature = "gcp")]
pub use object_store::gcp::GcpCredential;
use polars_core::config;
use polars_error::{PolarsResult, polars_bail};
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "python")]
use polars_utils::python_function::PythonObject;
#[cfg(feature = "python")]
use python_impl::PythonCredentialProvider;

#[derive(Clone, Debug, PartialEq, Hash, Eq)]
pub enum PlCredentialProvider {
    /// Prefer using [`PlCredentialProvider::from_func`] instead of constructing this directly
    Function(CredentialProviderFunction),
    #[cfg(feature = "python")]
    Python(PythonCredentialProvider),
}

impl PlCredentialProvider {
    /// Accepts a function that returns (credential, expiry time as seconds since UNIX_EPOCH)
    ///
    /// This functionality is unstable.
    pub fn from_func(
        // Internal notes
        // * This function is exposed as the Rust API for `PlCredentialProvider`
        func: impl Fn() -> Pin<
            Box<dyn Future<Output = PolarsResult<(ObjectStoreCredential, u64)>> + Send + Sync>,
        > + Send
        + Sync
        + 'static,
    ) -> Self {
        Self::Function(CredentialProviderFunction(Arc::new(func)))
    }

    /// Intended to be called with an internal `CredentialProviderBuilder` from
    /// py-polars.
    #[cfg(feature = "python")]
    pub fn from_python_builder(func: pyo3::Py<pyo3::PyAny>) -> Self {
        Self::Python(python_impl::PythonCredentialProvider::Builder(Arc::new(
            PythonObject(func),
        )))
    }

    pub(super) fn func_addr(&self) -> usize {
        match self {
            Self::Function(CredentialProviderFunction(v)) => Arc::as_ptr(v) as *const () as usize,
            #[cfg(feature = "python")]
            Self::Python(v) => v.func_addr(),
        }
    }

    /// Python passes a `CredentialProviderBuilder`, this calls the builder to build the final
    /// credential provider.
    ///
    /// This returns `Option` as the auto-initialization case is fallible and falls back to None.
    pub(crate) fn try_into_initialized(
        self,
        clear_cached_credentials: bool,
    ) -> PolarsResult<Option<Self>> {
        match self {
            Self::Function(_) => Ok(Some(self)),
            #[cfg(feature = "python")]
            Self::Python(v) => Ok(v
                .try_into_initialized(clear_cached_credentials)?
                .map(Self::Python)),
        }
    }
}

pub enum ObjectStoreCredential {
    #[cfg(feature = "aws")]
    Aws(Arc<object_store::aws::AwsCredential>),
    #[cfg(feature = "azure")]
    Azure(Arc<object_store::azure::AzureCredential>),
    #[cfg(feature = "gcp")]
    Gcp(Arc<object_store::gcp::GcpCredential>),
    /// For testing purposes
    None,
}

impl ObjectStoreCredential {
    fn variant_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "aws")]
            Self::Aws(_) => "Aws",
            #[cfg(feature = "azure")]
            Self::Azure(_) => "Azure",
            #[cfg(feature = "gcp")]
            Self::Gcp(_) => "Gcp",
            Self::None => "None",
        }
    }

    fn panic_type_mismatch(&self, expected: &str) {
        panic!(
            "impl error: credential type mismatch: expected {}, got {} instead",
            expected,
            self.variant_name()
        )
    }

    #[cfg(feature = "aws")]
    fn unwrap_aws(self) -> Arc<object_store::aws::AwsCredential> {
        let Self::Aws(v) = self else {
            self.panic_type_mismatch("aws");
            unreachable!()
        };
        v
    }

    #[cfg(feature = "azure")]
    fn unwrap_azure(self) -> Arc<object_store::azure::AzureCredential> {
        let Self::Azure(v) = self else {
            self.panic_type_mismatch("azure");
            unreachable!()
        };
        v
    }

    #[cfg(feature = "gcp")]
    fn unwrap_gcp(self) -> Arc<object_store::gcp::GcpCredential> {
        let Self::Gcp(v) = self else {
            self.panic_type_mismatch("gcp");
            unreachable!()
        };
        v
    }
}

pub trait IntoCredentialProvider: Sized {
    #[cfg(feature = "aws")]
    fn into_aws_provider(self) -> object_store::aws::AwsCredentialProvider {
        unimplemented!()
    }

    #[cfg(feature = "azure")]
    fn into_azure_provider(self) -> object_store::azure::AzureCredentialProvider {
        unimplemented!()
    }

    #[cfg(feature = "gcp")]
    fn into_gcp_provider(self) -> object_store::gcp::GcpCredentialProvider {
        unimplemented!()
    }

    /// Note, technically shouldn't be under the `IntoCredentialProvider` trait, but it's here
    /// for convenience.
    fn storage_update_options(&self) -> PolarsResult<Vec<(PlSmallStr, PlSmallStr)>>;
}

impl IntoCredentialProvider for PlCredentialProvider {
    #[cfg(feature = "aws")]
    fn into_aws_provider(self) -> object_store::aws::AwsCredentialProvider {
        match self {
            Self::Function(v) => v.into_aws_provider(),
            #[cfg(feature = "python")]
            Self::Python(v) => v.into_aws_provider(),
        }
    }

    #[cfg(feature = "azure")]
    fn into_azure_provider(self) -> object_store::azure::AzureCredentialProvider {
        match self {
            Self::Function(v) => v.into_azure_provider(),
            #[cfg(feature = "python")]
            Self::Python(v) => v.into_azure_provider(),
        }
    }

    #[cfg(feature = "gcp")]
    fn into_gcp_provider(self) -> object_store::gcp::GcpCredentialProvider {
        match self {
            Self::Function(v) => v.into_gcp_provider(),
            #[cfg(feature = "python")]
            Self::Python(v) => v.into_gcp_provider(),
        }
    }

    fn storage_update_options(&self) -> PolarsResult<Vec<(PlSmallStr, PlSmallStr)>> {
        match self {
            Self::Function(v) => v.storage_update_options(),
            #[cfg(feature = "python")]
            Self::Python(v) => v.storage_update_options(),
        }
    }
}

type CredentialProviderFunctionImpl = Arc<
    dyn Fn() -> Pin<
            Box<dyn Future<Output = PolarsResult<(ObjectStoreCredential, u64)>> + Send + Sync>,
        > + Send
        + Sync,
>;

/// Wrapper that implements [`IntoCredentialProvider`], [`Debug`], [`PartialEq`], [`Hash`] etc.
#[derive(Clone)]
pub struct CredentialProviderFunction(CredentialProviderFunctionImpl);

macro_rules! build_to_object_store_err {
    ($s:expr) => {{
        fn to_object_store_err(
            e: impl std::error::Error + Send + Sync + 'static,
        ) -> object_store::Error {
            object_store::Error::Generic {
                store: $s,
                source: Box::new(e),
            }
        }

        to_object_store_err
    }};
}

impl IntoCredentialProvider for CredentialProviderFunction {
    #[cfg(feature = "aws")]
    fn into_aws_provider(self) -> object_store::aws::AwsCredentialProvider {
        #[derive(Debug)]
        struct S(
            CredentialProviderFunction,
            FetchedCredentialsCache<Arc<object_store::aws::AwsCredential>>,
        );

        #[async_trait]
        impl object_store::CredentialProvider for S {
            type Credential = object_store::aws::AwsCredential;

            async fn get_credential(&self) -> object_store::Result<Arc<Self::Credential>> {
                self.1
                    .get_maybe_update(async {
                        let (creds, expiry) = self.0.0().await?;
                        PolarsResult::Ok((creds.unwrap_aws(), expiry))
                    })
                    .await
                    .map_err(build_to_object_store_err!("credential-provider-aws"))
            }
        }

        Arc::new(S(
            self,
            FetchedCredentialsCache::new(Arc::new(AwsCredential {
                key_id: String::new(),
                secret_key: String::new(),
                token: None,
            })),
        ))
    }

    #[cfg(feature = "azure")]
    fn into_azure_provider(self) -> object_store::azure::AzureCredentialProvider {
        #[derive(Debug)]
        struct S(
            CredentialProviderFunction,
            FetchedCredentialsCache<Arc<object_store::azure::AzureCredential>>,
        );

        #[async_trait]
        impl object_store::CredentialProvider for S {
            type Credential = object_store::azure::AzureCredential;

            async fn get_credential(&self) -> object_store::Result<Arc<Self::Credential>> {
                self.1
                    .get_maybe_update(async {
                        let (creds, expiry) = self.0.0().await?;
                        PolarsResult::Ok((creds.unwrap_azure(), expiry))
                    })
                    .await
                    .map_err(build_to_object_store_err!("credential-provider-azure"))
            }
        }

        Arc::new(S(
            self,
            FetchedCredentialsCache::new(Arc::new(AzureCredential::BearerToken(String::new()))),
        ))
    }

    #[cfg(feature = "gcp")]
    fn into_gcp_provider(self) -> object_store::gcp::GcpCredentialProvider {
        #[derive(Debug)]
        struct S(
            CredentialProviderFunction,
            FetchedCredentialsCache<Arc<object_store::gcp::GcpCredential>>,
        );

        #[async_trait]
        impl object_store::CredentialProvider for S {
            type Credential = object_store::gcp::GcpCredential;

            async fn get_credential(&self) -> object_store::Result<Arc<Self::Credential>> {
                self.1
                    .get_maybe_update(async {
                        let (creds, expiry) = self.0.0().await?;
                        PolarsResult::Ok((creds.unwrap_gcp(), expiry))
                    })
                    .await
                    .map_err(build_to_object_store_err!("credential-provider-gcp"))
            }
        }

        Arc::new(S(
            self,
            FetchedCredentialsCache::new(Arc::new(GcpCredential {
                bearer: String::new(),
            })),
        ))
    }

    fn storage_update_options(&self) -> PolarsResult<Vec<(PlSmallStr, PlSmallStr)>> {
        Ok(vec![])
    }
}

impl Debug for CredentialProviderFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "credential provider function at 0x{:016x}",
            self.0.as_ref() as *const _ as *const () as usize
        )
    }
}

impl Eq for CredentialProviderFunction {}

impl PartialEq for CredentialProviderFunction {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Hash for CredentialProviderFunction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(Arc::as_ptr(&self.0) as *const () as usize)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for PlCredentialProvider {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[cfg(feature = "python")]
        {
            Ok(Self::Python(PythonCredentialProvider::deserialize(
                _deserializer,
            )?))
        }
        #[cfg(not(feature = "python"))]
        {
            use serde::de::Error;
            Err(D::Error::custom("cannot deserialize PlCredentialProvider"))
        }
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for PlCredentialProvider {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;

        #[cfg(feature = "python")]
        if let PlCredentialProvider::Python(v) = self {
            return v.serialize(_serializer);
        }

        Err(S::Error::custom(format!("cannot serialize {self:?}")))
    }
}

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for PlCredentialProvider {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "PlCredentialProvider".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "PlCredentialProvider"))
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Vec::<u8>::json_schema(generator)
    }
}

/// Avoids calling the credential provider function if we have not yet passed the expiry time.
#[derive(Debug)]
struct FetchedCredentialsCache<C>(tokio::sync::Mutex<(C, u64, bool)>);

impl<C: Clone> FetchedCredentialsCache<C> {
    fn new(init_creds: C) -> Self {
        Self(tokio::sync::Mutex::new((init_creds, 0, true)))
    }

    async fn get_maybe_update(
        &self,
        // Taking an `impl Future` here allows us to potentially avoid a `Box::pin` allocation from
        // a `Fn() -> Pin<Box<dyn Future>>` by having it wrapped in an `async { f() }` block. We
        // will not poll that block if the credentials have not yet expired.
        update_func: impl Future<Output = PolarsResult<(C, u64)>>,
    ) -> PolarsResult<C> {
        let verbose = config::verbose();

        fn expiry_msg(last_fetched_expiry: u64, now: u64) -> String {
            if last_fetched_expiry == u64::MAX {
                "expiry = (never expires)".into()
            } else {
                format!(
                    "expiry = {} (in {} seconds)",
                    last_fetched_expiry,
                    last_fetched_expiry.saturating_sub(now)
                )
            }
        }

        let mut inner = self.0.lock().await;
        let (last_fetched_credentials, last_fetched_expiry, log_use_cached) = &mut *inner;

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if *last_fetched_expiry <= current_time {
            if verbose {
                eprintln!(
                    "[FetchedCredentialsCache]: \
                    Call update_func: current_time = {}, \
                    last_fetched_expiry = {}",
                    current_time, *last_fetched_expiry
                )
            }

            let (credentials, expiry) = update_func.await?;

            *last_fetched_credentials = credentials;
            *last_fetched_expiry = expiry;
            *log_use_cached = true;

            if expiry < current_time && expiry != 0 {
                polars_bail!(
                    ComputeError:
                    "credential expiry time {} is older than system time {} \
                     by {} seconds",
                    expiry,
                    current_time,
                    current_time - expiry
                )
            }

            if verbose {
                eprintln!(
                    "[FetchedCredentialsCache]: Finish update_func: new {}",
                    expiry_msg(
                        *last_fetched_expiry,
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    )
                )
            }
        } else if verbose && *log_use_cached {
            *log_use_cached = false;
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            eprintln!(
                "[FetchedCredentialsCache]: Using cached credentials: \
                current_time = {}, {}",
                now,
                expiry_msg(*last_fetched_expiry, now)
            )
        }

        Ok(last_fetched_credentials.clone())
    }
}

#[cfg(feature = "python")]
mod python_impl {
    use std::hash::Hash;
    use std::sync::Arc;

    use polars_error::{PolarsError, PolarsResult};
    use polars_utils::pl_str::PlSmallStr;
    use polars_utils::python_function::PythonObject;
    use pyo3::exceptions::PyValueError;
    use pyo3::pybacked::PyBackedStr;
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};
    use pyo3::{Python, intern};

    use super::IntoCredentialProvider;

    #[derive(Clone, Debug)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub enum PythonCredentialProvider {
        #[cfg_attr(
            feature = "serde",
            serde(
                serialize_with = "PythonObject::serialize_with_pyversion",
                deserialize_with = "PythonObject::deserialize_with_pyversion"
            )
        )]
        /// Indicates `py_object` is a `CredentialProviderBuilder`.
        Builder(Arc<PythonObject>),
        #[cfg_attr(
            feature = "serde",
            serde(
                serialize_with = "PythonObject::serialize_with_pyversion",
                deserialize_with = "PythonObject::deserialize_with_pyversion"
            )
        )]
        /// Indicates `py_object` is an instantiated credential provider
        Provider(Arc<PythonObject>),
    }

    impl PythonCredentialProvider {
        /// Performs initialization if necessary.
        ///
        /// This exists as a separate step that must be called beforehand. This approach is easier
        /// as the alternative is to refactor the `IntoCredentialProvider` trait to return
        /// `PolarsResult<Option<T>>` for every single function.
        pub(super) fn try_into_initialized(
            self,
            clear_cached_credentials: bool,
        ) -> PolarsResult<Option<Self>> {
            match self {
                Self::Builder(py_object) => {
                    let opt_initialized_py_object = Python::attach(|py| {
                        let build_fn =
                            py_object.getattr(py, intern!(py, "build_credential_provider"))?;

                        let v = build_fn.call1(py, (clear_cached_credentials,))?;
                        let v = (!v.is_none(py)).then_some(v);

                        pyo3::PyResult::Ok(v)
                    })?;

                    Ok(opt_initialized_py_object
                        .map(PythonObject)
                        .map(Arc::new)
                        .map(Self::Provider))
                },
                Self::Provider(_) => {
                    // Note: We don't expect to hit here.
                    Ok(Some(self))
                },
            }
        }

        fn unwrap_as_provider(self) -> Arc<PythonObject> {
            match self {
                Self::Builder(_) => panic!(),
                Self::Provider(v) => v,
            }
        }

        pub(crate) fn unwrap_as_provider_ref(&self) -> &Arc<PythonObject> {
            match self {
                Self::Builder(_) => panic!(),
                Self::Provider(v) => v,
            }
        }

        pub(super) fn func_addr(&self) -> usize {
            (match self {
                Self::Builder(v) => Arc::as_ptr(v),
                Self::Provider(v) => Arc::as_ptr(v),
            }) as *const () as usize
        }
    }

    impl IntoCredentialProvider for PythonCredentialProvider {
        #[cfg(feature = "aws")]
        fn into_aws_provider(self) -> object_store::aws::AwsCredentialProvider {
            use polars_error::PolarsResult;

            use crate::cloud::credential_provider::{
                CredentialProviderFunction, ObjectStoreCredential,
            };

            let func = self.unwrap_as_provider();

            CredentialProviderFunction(Arc::new(move || {
                let func = func.clone();
                Box::pin(async move {
                    let mut credentials = object_store::aws::AwsCredential {
                        key_id: String::new(),
                        secret_key: String::new(),
                        token: None,
                    };

                    let expiry = Python::attach(|py| {
                        let v = func.0.call0(py)?.into_bound(py);
                        let (storage_options, expiry) =
                            v.extract::<(pyo3::Bound<'_, PyDict>, Option<u64>)>()?;

                        for (k, v) in storage_options.iter() {
                            let k = k.extract::<PyBackedStr>()?;
                            let v = v.extract::<Option<String>>()?;

                            match k.as_ref() {
                                "aws_access_key_id" => {
                                    credentials.key_id = v.ok_or_else(|| {
                                        PyValueError::new_err("aws_access_key_id was None")
                                    })?;
                                },
                                "aws_secret_access_key" => {
                                    credentials.secret_key = v.ok_or_else(|| {
                                        PyValueError::new_err("aws_secret_access_key was None")
                                    })?
                                },
                                "aws_session_token" => credentials.token = v,
                                v => {
                                    return pyo3::PyResult::Err(PyValueError::new_err(format!(
                                        "unknown configuration key for aws: {}, \
                                    valid configuration keys are: \
                                    {}, {}, {}",
                                        v,
                                        "aws_access_key_id",
                                        "aws_secret_access_key",
                                        "aws_session_token"
                                    )));
                                },
                            }
                        }

                        pyo3::PyResult::Ok(expiry.unwrap_or(u64::MAX))
                    })?;

                    if credentials.key_id.is_empty() {
                        return Err(PolarsError::ComputeError(
                            "aws_access_key_id was empty or not given".into(),
                        ));
                    }

                    if credentials.secret_key.is_empty() {
                        return Err(PolarsError::ComputeError(
                            "aws_secret_access_key was empty or not given".into(),
                        ));
                    }

                    PolarsResult::Ok((ObjectStoreCredential::Aws(Arc::new(credentials)), expiry))
                })
            }))
            .into_aws_provider()
        }

        #[cfg(feature = "azure")]
        fn into_azure_provider(self) -> object_store::azure::AzureCredentialProvider {
            use object_store::azure::AzureAccessKey;
            use percent_encoding::percent_decode_str;
            use polars_core::config::verbose_print_sensitive;
            use polars_error::PolarsResult;

            use crate::cloud::credential_provider::{
                CredentialProviderFunction, ObjectStoreCredential,
            };

            let func = self.unwrap_as_provider();

            return CredentialProviderFunction(Arc::new(move || {
                let func = func.clone();
                Box::pin(async move {
                    let mut credentials = None;

                    static VALID_KEYS_MSG: &str =
                        "valid configuration keys are: ('account_key', 'bearer_token', 'sas_token')";

                    let expiry = Python::attach(|py| {
                        let v = func.0.call0(py)?.into_bound(py);
                        let (storage_options, expiry) =
                            v.extract::<(pyo3::Bound<'_, PyDict>, Option<u64>)>()?;

                        for (k, v) in storage_options.iter() {
                            let k = k.extract::<PyBackedStr>()?;
                            let v = v.extract::<String>()?;

                            match k.as_ref() {
                                "account_key" => {
                                    credentials =
                                        Some(object_store::azure::AzureCredential::AccessKey(
                                            AzureAccessKey::try_new(v.as_str()).map_err(|e| {
                                                PyValueError::new_err(e.to_string())
                                            })?,
                                        ))
                                },
                                "bearer_token" => {
                                    credentials =
                                        Some(object_store::azure::AzureCredential::BearerToken(v))
                                },
                                "sas_token" => {
                                    credentials =
                                        Some(object_store::azure::AzureCredential::SASToken(
                                            split_sas(&v).map_err(|err_msg| {
                                                verbose_print_sensitive(|| {
                                                    format!("error decoding SAS token: {err_msg} (token: {v})")
                                                });

                                                PyValueError::new_err(format!(
                                                    "error decoding SAS token: {err_msg}. \
                                                    Set POLARS_VERBOSE_SENSITIVE=1 to print the value"
                                                ))
                                            })?,
                                        ))
                                },
                                v => {
                                    return pyo3::PyResult::Err(PyValueError::new_err(format!(
                                        "unknown configuration key for azure: {v}, {VALID_KEYS_MSG}"
                                    )));
                                },
                            }
                        }

                        pyo3::PyResult::Ok(expiry.unwrap_or(u64::MAX))
                    })?;

                    let Some(credentials) = credentials else {
                        return Err(PolarsError::ComputeError(
                            format!(
                                "did not find a valid configuration key for azure, {VALID_KEYS_MSG}"
                            )
                            .into(),
                        ));
                    };

                    PolarsResult::Ok((ObjectStoreCredential::Azure(Arc::new(credentials)), expiry))
                })
            }))
            .into_azure_provider();

            /// Copied and adjusted from object-store.
            ///
            /// https://github.com/apache/arrow-rs-object-store/blob/7a0504b4924fcecee17d768fd7190b8f71b0877f/src/azure/builder.rs#L1072-L1089
            fn split_sas(sas: &str) -> Result<Vec<(String, String)>, &'static str> {
                let sas = percent_decode_str(sas)
                    .decode_utf8()
                    .map_err(|_| "UTF-8 decode error")?;

                let kv_str_pairs = sas
                    .trim_start_matches('?')
                    .split('&')
                    .filter(|s| !s.chars().all(char::is_whitespace));

                let mut pairs = Vec::new();

                for kv_pair_str in kv_str_pairs {
                    let (k, v) = kv_pair_str
                        .trim()
                        .split_once('=')
                        .ok_or("missing SAS component")?;
                    pairs.push((k.into(), v.into()))
                }

                Ok(pairs)
            }
        }

        #[cfg(feature = "gcp")]
        fn into_gcp_provider(self) -> object_store::gcp::GcpCredentialProvider {
            use polars_error::PolarsResult;

            use crate::cloud::credential_provider::{
                CredentialProviderFunction, ObjectStoreCredential,
            };

            let func = self.unwrap_as_provider();

            CredentialProviderFunction(Arc::new(move || {
                let func = func.clone();
                Box::pin(async move {
                    let mut credentials = object_store::gcp::GcpCredential {
                        bearer: String::new(),
                    };

                    let expiry = Python::attach(|py| {
                        let v = func.0.call0(py)?.into_bound(py);
                        let (storage_options, expiry) =
                            v.extract::<(pyo3::Bound<'_, PyDict>, Option<u64>)>()?;

                        for (k, v) in storage_options.iter() {
                            let k = k.extract::<PyBackedStr>()?;
                            let v = v.extract::<String>()?;

                            match k.as_ref() {
                                "bearer_token" => credentials.bearer = v,
                                v => {
                                    return pyo3::PyResult::Err(PyValueError::new_err(format!(
                                        "unknown configuration key for gcp: {}, \
                                    valid configuration keys are: {}",
                                        v, "bearer_token",
                                    )));
                                },
                            }
                        }

                        pyo3::PyResult::Ok(expiry.unwrap_or(u64::MAX))
                    })?;

                    if credentials.bearer.is_empty() {
                        return Err(PolarsError::ComputeError(
                            "bearer was empty or not given".into(),
                        ));
                    }

                    PolarsResult::Ok((ObjectStoreCredential::Gcp(Arc::new(credentials)), expiry))
                })
            }))
            .into_gcp_provider()
        }

        /// # Panics
        /// Panics if `self` is not an initialized provider.
        fn storage_update_options(&self) -> PolarsResult<Vec<(PlSmallStr, PlSmallStr)>> {
            let py_object = self.unwrap_as_provider_ref();

            Python::attach(|py| {
                py_object
                    .getattr(py, "_storage_update_options")
                    .map_or(Ok(vec![]), |f| {
                        let v = f
                            .call0(py)?
                            .extract::<pyo3::Bound<'_, PyDict>>(py)
                            .map_err(pyo3::PyErr::from)?;

                        let mut out = Vec::with_capacity(v.len());

                        for dict_item in v.call_method0("items")?.try_iter()? {
                            let (key, value) =
                                dict_item?.extract::<(PyBackedStr, PyBackedStr)>()?;

                            out.push(((&*key).into(), (&*value).into()))
                        }

                        Ok(out)
                    })
            })
        }
    }

    // Note: We don't consider `is_builder` for hash/eq - we don't expect the same Arc<PythonObject>
    // to be referenced as both true and false from the `is_builder` field.

    impl Eq for PythonCredentialProvider {}

    impl PartialEq for PythonCredentialProvider {
        fn eq(&self, other: &Self) -> bool {
            self.func_addr() == other.func_addr()
        }
    }

    impl Hash for PythonCredentialProvider {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            // # Safety
            // * Inner is an `Arc`
            // * Visibility is limited to super
            // * No code in `mod python_impl` or `super` mutates the Arc inner.
            state.write_usize(self.func_addr())
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "serde")]
    #[allow(clippy::redundant_pattern_matching)]
    #[test]
    fn test_serde() {
        use super::*;

        assert!(matches!(
            serde_json::to_string(&Some(PlCredentialProvider::from_func(|| {
                Box::pin(core::future::ready(PolarsResult::Ok((
                    ObjectStoreCredential::None,
                    0,
                ))))
            }))),
            Err(_)
        ));

        assert!(matches!(
            serde_json::to_string(&Option::<PlCredentialProvider>::None),
            Ok(String { .. })
        ));

        assert!(matches!(
            serde_json::from_str::<Option<PlCredentialProvider>>(
                serde_json::to_string(&Option::<PlCredentialProvider>::None)
                    .unwrap()
                    .as_str()
            ),
            Ok(None)
        ));
    }
}
