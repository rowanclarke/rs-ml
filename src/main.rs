mod activation;
mod layer;
mod loss;
mod matrix;
mod model;

use activation::sigmoid::Sigmoid;
use layer::feed::Feed;
use loss::mse::MeanSquaredError;
use model::{Model, ModelBuilder};
use ndarray::array;

fn main() {
    let mut model = ModelBuilder::new(vec![2]);
    model.push_layer(Box::new(Feed::<Sigmoid>::new(2)));
    model.push_layer(Box::new(Feed::<Sigmoid>::new(1)));

    let mut model = model.compile::<MeanSquaredError>(0.2);

    let inputs = [
        array![[0.0], [0.0]],
        array![[0.0], [1.0]],
        array![[1.0], [0.0]],
        array![[1.0], [1.0]],
    ];
    let targets = [array![[0.0]], array![[1.0]], array![[1.0]], array![[0.0]]];

    model.test(&inputs);
    model.train(&inputs, &targets, 10000);
    model.test(&inputs);
}
