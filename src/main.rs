mod activation;
mod layer;
mod model;

use activation::sigmoid::Sigmoid;
use layer::{feed::Feed, parallel::Parallel};
use model::Model;

fn main() {
    let mut model = Model::new(2);
    model.push_layer::<Feed<Sigmoid>>(2);
    model.push_layer::<Feed<Sigmoid>>(1);

    let inputs = &[
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = &[vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    model.train(inputs, targets, 10000, 0.5);
}
