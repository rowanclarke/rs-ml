use super::super::matrix::Column;
use super::{Loss, LossBuilder};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct MeanSquaredError {}

#[typetag::serde]
impl Loss for MeanSquaredError {
    fn forward(&self, output: Column, target: Column) -> f32 {
        let mut sum = 0.0;
        for i in 0..output.len() {
            sum += (output[i] - target[i]).powf(2.0);
        }
        sum
    }

    fn backward(&self, output: Column, target: Column) -> Column {
        let mut result = Column::zeros(output.len());
        for i in 0..output.len() {
            result[i] = 2.0 * (output[i] - target[i]);
        }
        result
    }
}

impl LossBuilder for MeanSquaredError {
    fn new() -> Self {
        MeanSquaredError {}
    }
}
