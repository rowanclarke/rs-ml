mod activation;
mod layer;
mod model;

use activation::sigmoid::Sigmoid;
use layer::{conv2d::TempConv2D, feed::Feed, Fixed, Layer};
use model::Model;

fn main() {
    let mut model = Model::new(vec![64, 64]);
    model.push_layer(TempConv2D {
        filters: 1,
        kernel_size: (3, 3),
    });
}
