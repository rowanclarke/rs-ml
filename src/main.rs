mod activation;
mod layer;

use activation::sigmoid::Sigmoid;
use layer::{feed::Feed, parallel::Parallel, series::Series, Layer};

fn main() {
    let mut series = Series::new();
    series.push(Box::new(Feed::<Sigmoid>::new(2, 3)));
}
