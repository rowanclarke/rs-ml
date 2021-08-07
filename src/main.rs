mod activation;
mod layer;
mod loss;
mod matrix;
mod model;

use activation::{relu::ReLU, softmax::Softmax};
use layer::{
    conv2d::{Conv2D, MaxPooling2D},
    feed::Feed,
    reshape::Flatten,
};
use loss::xent::CrossEntropy;
use matrix::Column;
use mnist::{Mnist, MnistBuilder};
use model::ModelBuilder;
use ndarray::{s, Array1, Array2, Array4};

fn main() {
    let trn_size: usize = 100;
    let tst_size: usize = 20;

    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size as u32)
        .test_set_length(tst_size as u32)
        .label_format_one_hot()
        .finalize();

    let trn_img = Array4::<u8>::from_shape_vec((trn_size, 28, 28, 1), trn_img)
        .unwrap()
        .map(|&x| x as f32 / 255.0);

    let trn_lbl = Array2::<u8>::from_shape_vec((trn_size, 10), trn_lbl)
        .unwrap()
        .map(|&x| x as f32);

    let mut _trn_img = Vec::<Column>::with_capacity(trn_size);
    let mut _trn_lbl = Vec::<Column>::with_capacity(trn_size);

    for i in 0..trn_size {
        let img = trn_img.slice(s![i, .., .., ..]).to_owned();
        _trn_img.push(Column::from_arr(img));

        let lbl = trn_lbl.slice(s![i, ..]).to_owned();
        _trn_lbl.push(Column::from_arr(lbl));
    }

    let trn_img = _trn_img.as_slice();
    let trn_lbl = _trn_lbl.as_slice();

    let mut model = ModelBuilder::new(vec![28, 28, 1]);
    model.push_layer(Conv2D::<ReLU>::new(32, (3, 3)));
    model.push_layer(MaxPooling2D::new((2, 2)));
    model.push_layer(Conv2D::<ReLU>::new(64, (4, 4)));
    model.push_layer(MaxPooling2D::new((2, 2)));
    model.push_layer(Flatten::new());
    model.push_layer(Feed::<Softmax>::new(10));

    let mut model = model.compile::<CrossEntropy>(0.1);
    model.train(trn_img, trn_lbl, 1);
}
