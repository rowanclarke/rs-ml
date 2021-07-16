mod activation;
mod layer;
mod model;

use activation::sigmoid::Sigmoid;
use layer::{conv2d::Conv2D, feed::Feed, Layer};
use model::Model;

fn main() {
    let mut model = Model::new(vec![64, 64]);
    model.push_layer::<Conv2D>(vec![5, 5]);
    model.push_layer::<Conv2D>(vec![5, 5]);

    let inputs = &[
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = &[vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    model.train(inputs, targets, 10001, 0.1);
}
