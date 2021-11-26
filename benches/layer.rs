use ml::activation::{relu::ReLU, Activation};
use ml::array::{Array2, Array3, Array4};
use ml::layer::{
    conv2d::{Conv2D, Conv2DLayer},
    Layer, LayerBuilder,
};
use ml::matrix::Column;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    let conv = Conv2D::<ReLU>::new(8, (3, 3));
    let mut conv = conv.build(vec![28, 28, 3]);
    c.bench_function("conv2d (28x28x3 -> 26x26x8)", |b| {
        b.iter(|| conv.forward(Column::random(28 * 28 * 3)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
