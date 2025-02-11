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

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use arrow_array::{cast::AsArray, types::Float32Type, Array, RecordBatch};
use arrow_schema::Field;
use async_trait::async_trait;
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result};
use lance_linalg::MatrixView;

use super::ProductQuantizer;
use crate::vector::transform::Transformer;

/// Product Quantizer Transformer
///
/// It transforms a column of vectors into a column of PQ codes.
pub struct PQTransformer {
    quantizer: Arc<ProductQuantizer>,
    input_column: String,
    output_column: String,
}

impl PQTransformer {
    pub fn new(quantizer: Arc<ProductQuantizer>, input_column: &str, output_column: &str) -> Self {
        Self {
            quantizer,
            input_column: input_column.to_owned(),
            output_column: output_column.to_owned(),
        }
    }
}

impl Debug for PQTransformer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PQTransformer(input={}, output={})",
            self.input_column, self.output_column
        )
    }
}

#[async_trait]
impl Transformer for PQTransformer {
    async fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let input_arr = batch
            .column_by_name(&self.input_column)
            .ok_or(Error::Index {
                message: format!(
                    "PQ Transform: column {} not found in batch",
                    self.input_column
                ),
            })?;
        let data: MatrixView<Float32Type> = input_arr
            .as_fixed_size_list_opt()
            .ok_or(Error::Index {
                message: format!(
                    "PQ Transform: column {} is not a fixed size list, got {}",
                    self.input_column,
                    input_arr.data_type(),
                ),
            })?
            .try_into()?;
        let pq_code = self.quantizer.transform(&data).await?;
        let pq_field = Field::new(&self.output_column, pq_code.data_type().clone(), false);
        let batch = batch.try_with_column(pq_field, Arc::new(pq_code))?;
        let batch = batch.drop_column(&self.input_column)?;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{FixedSizeListArray, Float32Array, Int32Array};
    use arrow_schema::{DataType, Schema};
    use lance_arrow::FixedSizeListArrayExt;

    use crate::vector::pq::PQBuildParams;

    #[tokio::test]
    async fn test_pq_transform() {
        let values = Float32Array::from_iter((0..16000).map(|v| v as f32));
        let dim = 16;
        let arr = Arc::new(FixedSizeListArray::try_new_from_values(values, 16).unwrap());
        let mat: MatrixView<Float32Type> = arr.as_ref().try_into().unwrap();

        let params = PQBuildParams::new(1, 8);
        let pq = ProductQuantizer::train(&mat, &params).await.unwrap();

        let schema = Schema::new(vec![
            Field::new(
                "vec",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
                true,
            ),
            Field::new("other", DataType::Int32, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![arr, Arc::new(Int32Array::from_iter_values(0..1000))],
        )
        .unwrap();

        let transformer = PQTransformer::new(pq.into(), "vec", "pq_code");
        let batch = transformer.transform(&batch).await.unwrap();
        assert!(batch.column_by_name("vec").is_none());
        assert!(batch.column_by_name("pq_code").is_some());
        assert!(batch.column_by_name("other").is_some());
        assert_eq!(batch.num_rows(), 1000)
    }
}
