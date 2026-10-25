use polars_core::runtime::ASYNC;
use polars_error::PolarsResult;

use crate::cloud::ObjectStoreErrorContext;

/// Wrapper for [`object_store::MultipartUpload`] that handles error conversion.
pub struct PlMultipartUpload {
    inner: Option<Box<dyn object_store::MultipartUpload>>,
    error_cx: ObjectStoreErrorContext,
    completed: bool,
}

impl PlMultipartUpload {
    pub fn new(
        inner: Box<dyn object_store::MultipartUpload>,
        error_cx: ObjectStoreErrorContext,
    ) -> Self {
        Self {
            inner: Some(inner),
            error_cx,
            completed: false,
        }
    }

    pub fn put(
        &mut self,
        payload: object_store::PutPayload,
    ) -> impl Future<Output = PolarsResult<()>> + Send + 'static {
        let fut = self.inner.as_mut().unwrap().put_part(payload);
        let error_cx = self.error_cx.clone();

        async move { fut.await.map_err(|e| error_cx.attach_err_info(e).into()) }
    }

    pub async fn finish(&mut self) -> PolarsResult<object_store::PutResult> {
        let result = self
            .inner
            .as_mut()
            .unwrap()
            .complete()
            .await
            .map_err(|e| self.error_cx.clone().attach_err_info(e).into());
        self.completed = result.is_ok();
        result
    }
}

impl Drop for PlMultipartUpload {
    fn drop(&mut self) {
        if let Some(mut inner) = self.inner.take()
            && !self.completed
        {
            ASYNC.spawn(async move {
                let _ = inner.abort().await;
            });
        }
    }
}
