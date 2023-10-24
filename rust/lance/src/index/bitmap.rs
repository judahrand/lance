// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Vector Index for Fast Approximate Nearest Neighbor (ANN) Search
//!

use arrow_array::Float32Array;
use arrow_array::RecordBatch;
use futures::StreamExt;
use arrow_array::iterator::ArrayIter;

use tracing::instrument;

use super::IndexParams;
use snafu::{location, Location};
use log::info;
use std::any::Any;
use crate::Dataset;
use lance_core::datatypes::Field;
use crate::error::{Error, Result};
use arrow_schema::DataType;
use bitrush_index::{BitmapIndex, OZBCBitmap};

const INDEX_FILE_NAME: &str = "index.idx";

/// The parameters to build bitmap index.
#[derive(Debug, Clone)]
pub struct BitmapIndexParams {
}

impl BitmapIndexParams {
}

impl IndexParams for BitmapIndexParams {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn sanity_check<'a>(dataset: &'a Dataset, column: &str) -> Result<&'a Field> {
    let Some(field) = dataset.schema().field(column) else {
        return Err(Error::IO {
            message: format!(
                "Building index: column {} does not exist in dataset: {:?}",
                column, dataset
            ),
            location: location!(),
        });
    };
    if let DataType::FixedSizeList(elem_type, _) = field.data_type() {
        if !matches!(elem_type.data_type(), DataType::Float32) {
            return Err(
        Error::Index{message:
            format!("VectorIndex requires the column data type to be fixed size list of float32s, got {}",
            elem_type.data_type())});
        }
    } else {
        return Err(Error::Index {
            message: format!(
            "VectorIndex requires the column data type to be fixed size list of float32s, got {}",
            field.data_type()
        ),
        });
    }
    Ok(field)
}

/// Build a Bitmap Index
#[instrument(skip(dataset))]
pub(crate) async fn build_bitmap_index(
    dataset: &Dataset,
    column: &str,
    name: &str,
    uuid: &str,
    params: &BitmapIndexParams,
) -> Result<()> {
    info!(
        "Building bitmap index:",
    );

    let field = sanity_check(dataset, column)?;

    let build_options = bitrush_index::new_default_index_options::<u64>();
    let b_index = match BitmapIndex::<OZBCBitmap, u64>::new(build_options) {
        Ok(b_index) => b_index,
        Err(err) => panic!("Error occured creating bitmap index: {:?}", err)
    };

    // Transform data, compute residuals and sort by partition ids.
    let mut scanner = dataset.scan();
    scanner.batch_readahead(num_cpus::get() * 2);
    scanner.project(&[column])?;
    scanner.with_row_id();

    // Scan the dataset and compute residual, pq with with partition ID.
    // For now, it loads all data into memory.
    let stream = scanner.try_into_stream().await?;
    stream.map(|b| b?.column_by_name(column)?.as_any().downcast_ref::<Float32Array>().unwrap().iter())
        .collect::<Vec<RecordBatch>>()
        .await;

    Ok(())
}

// #[instrument(skip_all, fields(old_uuid = old_uuid.to_string(), new_uuid = new_uuid.to_string(), num_rows = mapping.len()))]
// pub(crate) async fn remap_bitmap_index(
//     dataset: Arc<Dataset>,
//     column: &str,
//     old_uuid: &Uuid,
//     new_uuid: &Uuid,
//     old_metadata: &crate::format::Index,
//     mapping: &HashMap<u64, Option<u64>>,
// ) -> Result<()> {
//     let old_index = open_index(dataset.clone(), column, &old_uuid.to_string()).await?;
//     old_index.check_can_remap()?;
//     let ivf_index: &IVFIndex =
//         old_index
//             .as_any()
//             .downcast_ref()
//             .ok_or_else(|| Error::NotSupported {
//                 source: "Only IVF indexes can be remapped currently".into(),
//             })?;

//     remap_index_file(
//         dataset.as_ref(),
//         &old_uuid.to_string(),
//         &new_uuid.to_string(),
//         old_metadata.dataset_version,
//         ivf_index,
//         mapping,
//         old_metadata.name.clone(),
//         column.to_string(),
//         // We can safely assume there are no transforms today.  We assert above that the
//         // top stage is IVF and IVF does not support transforms between IVF and PQ.  This
//         // will be fixed in the future.
//         vec![],
//     )
//     .await?;
//     Ok(())
// }

// /// Open the Bitmap index on dataset, specified by the `uuid`.
// #[instrument(level = "debug", skip(dataset))]
// pub(crate) async fn open_index(
//     dataset: Arc<Dataset>,
//     column: &str,
//     uuid: &str,
// ) -> Result<Arc<dyn BitmapIndex>> {
//     if let Some(index) = dataset.session.index_cache.get(uuid) {
//         return Ok(index);
//     }

//     let index_dir = dataset.indices_dir().child(uuid);
//     let index_file = index_dir.child(INDEX_FILE_NAME);

//     let object_store = dataset.object_store();
//     let reader: Arc<dyn Reader> = object_store.open(&index_file).await?.into();

//     let file_size = reader.size().await?;
//     let block_size = object_store.block_size();
//     let begin = if file_size < block_size {
//         0
//     } else {
//         file_size - block_size
//     };
//     let tail_bytes = reader.get_range(begin..file_size).await?;
//     let metadata_pos = read_metadata_offset(&tail_bytes)?;
//     let proto: pb::Index = if metadata_pos < file_size - tail_bytes.len() {
//         // We have not read the metadata bytes yet.
//         read_message(reader.as_ref(), metadata_pos).await?
//     } else {
//         let offset = tail_bytes.len() - (file_size - metadata_pos);
//         read_message_from_buf(&tail_bytes.slice(offset..))?
//     };

//     if proto.columns.len() != 1 {
//         return Err(Error::Index {
//             message: "VectorIndex only supports 1 column".to_string(),
//         });
//     }
//     assert_eq!(proto.index_type, pb::IndexType::Vector as i32);

//     let Some(idx_impl) = proto.implementation.as_ref() else {
//         return Err(Error::Index {
//             message: "Invalid protobuf for VectorIndex metadata".to_string(),
//         });
//     };

//     let pb::index::Implementation::VectorIndex(vec_idx) = idx_impl;

//     let metric_type = pb::VectorMetricType::try_from(vec_idx.metric_type)?.into();

//     let mut last_stage: Option<Arc<dyn VectorIndex>> = None;

//     for stg in vec_idx.stages.iter().rev() {
//         match stg.stage.as_ref() {
//             #[allow(unused_variables)]
//             Some(Stage::Transform(tf)) => {
//                 if last_stage.is_none() {
//                     return Err(Error::Index {
//                         message: format!("Invalid vector index stages: {:?}", vec_idx.stages),
//                     });
//                 }
//                 #[cfg(feature = "opq")]
//                 match tf.r#type() {
//                     pb::TransformType::Opq => {
//                         let opq = OptimizedProductQuantizer::load(
//                             reader.as_ref(),
//                             tf.position as usize,
//                             tf.shape
//                                 .iter()
//                                 .map(|s| *s as usize)
//                                 .collect::<Vec<_>>()
//                                 .as_slice(),
//                         )
//                         .await?;
//                         last_stage = Some(Arc::new(OPQIndex::new(
//                             last_stage.as_ref().unwrap().clone(),
//                             opq,
//                         )));
//                     }
//                 }
//             }
//             Some(Stage::Ivf(ivf_pb)) => {
//                 if last_stage.is_none() {
//                     return Err(Error::Index {
//                         message: format!("Invalid vector index stages: {:?}", vec_idx.stages),
//                     });
//                 }
//                 let ivf = Ivf::try_from(ivf_pb)?;
//                 last_stage = Some(Arc::new(IVFIndex::try_new(
//                     dataset.session.clone(),
//                     uuid,
//                     ivf,
//                     reader.clone(),
//                     last_stage.unwrap(),
//                     metric_type,
//                 )?));
//             }
//             Some(Stage::Pq(pq_proto)) => {
//                 if last_stage.is_some() {
//                     return Err(Error::Index {
//                         message: format!("Invalid vector index stages: {:?}", vec_idx.stages),
//                     });
//                 };
//                 let pq = Arc::new(ProductQuantizer::new(
//                     pq_proto.num_sub_vectors as usize,
//                     pq_proto.num_bits,
//                     pq_proto.dimension as usize,
//                     Arc::new(Float32Array::from_iter_values(
//                         pq_proto.codebook.iter().copied(),
//                     )),
//                     metric_type,
//                 ));
//                 last_stage = Some(Arc::new(PQIndex::new(pq, metric_type)));
//             }
//             Some(Stage::Diskann(diskann_proto)) => {
//                 if last_stage.is_some() {
//                     return Err(Error::Index {
//                         message: format!(
//                             "DiskANN should be the only stage, but we got stages: {:?}",
//                             vec_idx.stages
//                         ),
//                     });
//                 };
//                 let graph_path = index_dir.child(diskann_proto.filename.as_str());
//                 let diskann =
//                     Arc::new(DiskANNIndex::try_new(dataset.clone(), column, &graph_path).await?);
//                 last_stage = Some(diskann);
//             }
//             _ => {}
//         }
//     }

//     if last_stage.is_none() {
//         return Err(Error::Index {
//             message: format!("Invalid index stages: {:?}", vec_idx.stages),
//         });
//     }
//     let idx = last_stage.unwrap();
//     dataset.session.index_cache.insert(uuid, idx.clone());
//     Ok(idx)
// }
