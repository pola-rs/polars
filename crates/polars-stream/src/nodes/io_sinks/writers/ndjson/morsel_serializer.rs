use arrow::array::{Array, StructArray};
use arrow::datatypes::{ArrowDataType, Field as ArrowField};
use polars_core::frame::DataFrame;
use polars_core::prelude::CompatLevel;
use polars_error::PolarsResult;

use crate::async_executor::{self, TaskPriority};
use crate::async_primitives::connector;
use crate::nodes::io_sinks::components::par_utils::rechunk_par;
use crate::nodes::io_sinks::components::sink_morsel::{SinkMorsel, SinkMorselPermit};

pub struct MorselSerializerPipeline {
    pub morsel_rx: connector::Receiver<SinkMorsel>,
    pub filled_serializer_tx: tokio::sync::mpsc::Sender<(
        async_executor::AbortOnDropHandle<PolarsResult<MorselSerializer>>,
        SinkMorselPermit,
    )>,
    pub reuse_serializer_rx: tokio::sync::mpsc::Receiver<MorselSerializer>,
    pub max_serializers: usize,
    pub base_allocation_size: usize,
}

impl MorselSerializerPipeline {
    pub async fn run(self) {
        let MorselSerializerPipeline {
            mut morsel_rx,
            filled_serializer_tx,
            mut reuse_serializer_rx,
            max_serializers,
            base_allocation_size,
        } = self;

        let mut num_created_serializers: usize = 0;

        while let Ok(morsel) = morsel_rx.recv().await {
            let morsel_serializer: MorselSerializer =
                if let Ok(serializer) = reuse_serializer_rx.try_recv() {
                    serializer
                } else if num_created_serializers < max_serializers {
                    num_created_serializers += 1;
                    MorselSerializer {
                        serialized_data: vec![],
                        allocation_size: base_allocation_size,
                    }
                } else if let Some(serializer) = reuse_serializer_rx.recv().await {
                    serializer
                } else {
                    break;
                };

            let (df, morsel_permit) = morsel.into_inner();

            let handle = async_executor::AbortOnDropHandle::new(async_executor::spawn(
                TaskPriority::High,
                morsel_serializer.serialize_morsel(df),
            ));

            if filled_serializer_tx
                .send((handle, morsel_permit))
                .await
                .is_err()
            {
                break;
            }
        }
    }
}

pub struct MorselSerializer {
    pub serialized_data: Vec<u8>,
    allocation_size: usize,
}

impl MorselSerializer {
    pub async fn serialize_morsel(mut self, mut df: DataFrame) -> PolarsResult<Self> {
        let MorselSerializer {
            serialized_data,
            allocation_size,
        } = &mut self;

        rechunk_par(unsafe { df.columns_mut_retain_schema() }).await;

        serialized_data.clear();
        serialized_data.reserve_exact(*allocation_size);

        let height = df.height();

        let (arrays, fields): (Vec<Box<dyn Array>>, Vec<ArrowField>) = df
            .into_columns()
            .into_iter()
            .map(|c| {
                let name = c.name().clone();
                let arr = c.rechunk_to_arrow(CompatLevel::newest());

                let field = ArrowField::new(name, arr.dtype().clone(), true);

                (arr, field)
            })
            .unzip();

        let array = StructArray::new(ArrowDataType::Struct(fields), height, arrays, None);
        let serializer = polars_json::json::write::new_serializer(&array, 0, usize::MAX);

        serializer.serialize_json_lines_to_vec(serialized_data, height);
        *allocation_size = usize::max(*allocation_size, serialized_data.capacity());

        Ok(self)
    }
}
