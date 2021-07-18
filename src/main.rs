mod activation;
mod layer;
mod model;

use activation::{sigmoid::Sigmoid, softmax::Softmax};
use layer::{conv2d::Conv2D, conv2d::MaxPooling2D, feed::Feed, reshape::Flatten};
use mnist::{Mnist, MnistBuilder};
use model::Model;
use ndarray::{Array2, Array4};

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

    let trn_lbl = Array2::<u8>::from_shape_vec((trn_size, 10), trn_lbl);

    let mut model = Model::new(vec![28, 28, 1]);
    model.push_layer(Conv2D::new(32, (3, 3)));
    model.push_layer(MaxPooling2D::new((2, 2)));
    model.push_layer(Conv2D::new(64, (4, 4)));
    model.push_layer(MaxPooling2D::new((2, 2)));
    model.push_layer(Flatten::new());
    model.push_layer(Feed::<Softmax>::new(10));
}
