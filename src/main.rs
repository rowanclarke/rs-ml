mod activation;
mod layer;
mod model;

use activation::sigmoid::Sigmoid;
use layer::{feed::Feed, parallel::Parallel};
use model::Model;

fn main() {
    let mut model = Model::new(5);
    model.push_layer::<Feed<Sigmoid>>(7);
    model.push_group::<Parallel>();
    model.push_layer::<Feed<Sigmoid>>(4);
    model.push_layer::<Feed<Sigmoid>>(4);
    model.pop_group();
    model.push_layer::<Feed<Sigmoid>>(2);
}
