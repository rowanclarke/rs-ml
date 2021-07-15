mod activation;
mod layer;
mod model;

use activation::sigmoid::Sigmoid;
use layer::feed::Feed;
use model::Model;

fn main() {
    let mut model = Model::new(vec![2]);
    model.push_layer::<Feed<Sigmoid>>(vec![2]);
    model.push_layer::<Feed<Sigmoid>>(vec![1]);

    let inputs = &[
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = &[vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    model.train(inputs, targets, 10001, 0.1);
}
