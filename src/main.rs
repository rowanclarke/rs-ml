mod activation;
mod array;
mod layer;
mod loss;
mod matrix;
mod model;

use activation::{relu::ReLU, sigmoid::Sigmoid, softmax::Softmax};
use array::{Array, Array2, Array4, Shape};
use layer::{
    conv2d::{Conv2D, MaxPooling2D},
    feed::Feed,
    reshape::Flatten,
};
use loss::xent::CrossEntropy;
use matrix::Column;
use mnist::{Mnist, MnistBuilder};
use model::ModelBuilder;

fn slice<'a, T: Shape>(dat: Array<T>, size: usize) -> Box<[Column]> {
    let iter = dat.axis();
    let dat: Vec<Column> = iter.map(|x| Column::new(x.to_vec())).collect();
    dat.into_boxed_slice()
}

fn lbl<'a>(lbl: Vec<u8>, size: usize) -> Box<[Column]> {
    let lbl = Array2::from_shape_vec([size, 10], lbl.into_iter().map(|x| x as f32).collect());
    slice(lbl, size)
}

fn img<'a>(img: Vec<u8>, size: usize) -> Box<[Column]> {
    let img = Array4::from_shape_vec(
        [size, 28, 28, 1],
        img.into_iter().map(|x| x as f32 / 255.0).collect(),
    );
    slice(img, size)
}

fn main() {
    let trn_size: usize = 1;
    let tst_size: usize = 20;

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size as u32)
        .test_set_length(tst_size as u32)
        .label_format_one_hot()
        .finalize();

    let mut model = ModelBuilder::new(vec![28, 28, 1]);
    model.push_layer(Conv2D::<Sigmoid>::new(2, (3, 3)));
    model.push_layer(MaxPooling2D::new((2, 2)));
    model.push_layer(Conv2D::<Sigmoid>::new(4, (4, 4)));
    model.push_layer(MaxPooling2D::new((2, 2)));
    model.push_layer(Flatten::new());
    model.push_layer(Feed::<Softmax>::new(10));

    let mut model = model.compile::<CrossEntropy>(0.1);

    let trn_img = img(trn_img, trn_size);
    let trn_lbl = lbl(trn_lbl, trn_size);

    let tst_img = img(tst_img, tst_size);
    let tst_lbl = lbl(tst_lbl, tst_size);

    model.train(trn_img, trn_lbl, 3);

    let model = bincode::serialize(&model).unwrap();
    println!("{:?}", model);
}
