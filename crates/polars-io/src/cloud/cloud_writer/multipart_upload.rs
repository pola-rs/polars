use polars_error::PolarsResult;

use crate::cloud::ObjectStoreErrorContext;

/// Wrapper for [`object_store::MultipartUpload`] that handles error conversion.
pub struct PlMultipartUpload {
    inner: Box<dyn object_store::MultipartUpload>,
    error_cx: ObjectStoreErrorContext,
}

impl PlMultipartUpload {
    pub fn new(
        inner: Box<dyn object_store::MultipartUpload>,
        error_cx: ObjectStoreErrorContext,
    ) -> Self {
        Self { inner, error_cx }
    }

    pub fn put(
        &mut self,
        payload: object_store::PutPayload,
    ) -> impl Future<Output = PolarsResult<()>> + Send + 'static {
        let fut = self.inner.put_part(payload);
        let error_cx = self.error_cx.clone();

        async move { fut.await.map_err(|e| error_cx.attach_err_info(e).into()) }
    }

    pub async fn finish(&mut self) -> PolarsResult<object_store::PutResult> {
        self.inner
            .complete()
            .await
            .map_err(|e| self.error_cx.clone().attach_err_info(e).into())
    }
}
