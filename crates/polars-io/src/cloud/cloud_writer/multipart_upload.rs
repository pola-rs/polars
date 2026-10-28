use polars_async::primitives::wait_group::WaitToken;
use polars_core::runtime::ASYNC;
use polars_error::PolarsResult;

use crate::cloud::ObjectStoreErrorContext;
use crate::cloud::cloud_writer::{CLEANUP_TIMEOUT, hold};

/// Wrapper for [`object_store::MultipartUpload`] that handles error conversion.
pub struct PlMultipartUpload {
    inner: Option<Box<dyn object_store::MultipartUpload>>,
    error_cx: ObjectStoreErrorContext,
    completed: bool,
    /// Held by the abort.
    wait_token: Option<WaitToken>,
}

impl PlMultipartUpload {
    pub fn new(
        inner: Box<dyn object_store::MultipartUpload>,
        error_cx: ObjectStoreErrorContext,
        wait_token: Option<WaitToken>,
    ) -> Self {
        Self {
            inner: Some(inner),
            error_cx,
            completed: false,
            wait_token,
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

    /// Call only when no part upload is in flight, as such a part may be stored after the abort.
    pub async fn abort(mut self) {
        let inner = self.inner.take().unwrap();
        abort(inner, self.error_cx.clone()).await
    }
}

impl Drop for PlMultipartUpload {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take()
            && !self.completed
        {
            ASYNC.spawn(hold(
                self.wait_token.take(),
                abort(inner, self.error_cx.clone()),
            ));
        }
    }
}

async fn abort(
    mut inner: Box<dyn object_store::MultipartUpload>,
    error_cx: ObjectStoreErrorContext,
) {
    let result = tokio::time::timeout(CLEANUP_TIMEOUT, inner.abort()).await;
    if polars_core::config::verbose() {
        match result {
            Ok(Ok(())) => {},
            Ok(Err(e)) => eprintln!(
                "[CloudWriter]: failed to abort a multipart upload: {}",
                error_cx.attach_err_info(e)
            ),
            Err(_) => eprintln!("[CloudWriter]: timed out aborting a multipart upload"),
        }
    }
}
