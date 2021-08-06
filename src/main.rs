#![allow(warnings)]

mod activation;
mod layer;
mod loss;
mod matrix;
mod model;

use activation::sigmoid::Sigmoid;
use layer::feed::Feed;
use loss::mse::MeanSquaredError;
use matrix::Column;
use model::{Model, ModelBuilder};
use ndarray::array;

fn main() {
    let mut model = ModelBuilder::new(vec![2]);
    model.push_layer(Feed::<Sigmoid>::new(2));
    model.push_layer(Feed::<Sigmoid>::new(1));

    let mut model = model.compile::<MeanSquaredError>(0.2);

    let inputs = [
        Column::new(vec![0.0, 0.0]),
        Column::new(vec![0.0, 1.0]),
        Column::new(vec![1.0, 0.0]),
        Column::new(vec![1.0, 1.0]),
    ];
    let targets = [
        Column::new(vec![0.0]),
        Column::new(vec![1.0]),
        Column::new(vec![1.0]),
        Column::new(vec![0.0]),
    ];

    model.test(&inputs);
    model.train(&inputs, &targets, 10000);
    model.test(&inputs);
}
