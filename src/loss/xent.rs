use super::super::matrix::Column;
use super::{Loss, LossBuilder};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct CrossEntropy {}

#[typetag::serde]
impl Loss for CrossEntropy {
    fn forward(&self, output: Column, target: Column) -> f32 {
        let mut sum = 0.0;
        for i in 0..output.len() {
            sum += target[i] * output[i].ln();
        }
        -sum
    }

    fn backward(&self, output: Column, target: Column) -> Column {
        let mut del = Column::zeros(output.len());
        for i in 0..output.len() {
            del[i] = -target[i] * output[i].recip();
        }
        del
    }
}

impl LossBuilder for CrossEntropy {
    fn new() -> Self {
        CrossEntropy {}
    }
}
