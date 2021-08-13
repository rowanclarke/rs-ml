mod activation;
mod layer;
mod loss;
mod model;

use activation::{relu::ReLU, sigmoid::Sigmoid, softmax::Softmax};
use layer::{conv2d::Conv2D, conv2d::MaxPooling2D, feed::Feed, reshape::Flatten};
use loss::mse::MeanSquaredError;
use mnist::{Mnist, MnistBuilder};
<<<<<<< Updated upstream
use model::Model;
use ndarray::{Array2, Array4};
=======
use model::ModelBuilder;
use ndarray::{s, Array, Array2, Array4, Axis, RemoveAxis};

fn slice<'a, D: RemoveAxis>(dat: Array<f32, D>, size: usize) -> Box<[Column]> {
    let mut _dat = Vec::<Column>::with_capacity(size);

    for i in 0..size {
        let slice = dat.index_axis(Axis(0), i).to_owned();
        _dat.push(Column::from_arr(slice));
    }

    _dat.into_boxed_slice()
}

fn lbl<'a>(lbl: Vec<u8>, size: usize) -> Box<[Column]> {
    let lbl = Array2::<u8>::from_shape_vec((size, 10), lbl)
        .unwrap()
        .map(|&x| x as f32);
    slice(lbl, size)
}

fn img<'a>(img: Vec<u8>, size: usize) -> Box<[Column]> {
    let img = Array4::<u8>::from_shape_vec((size, 28, 28, 1), img)
        .unwrap()
        .map(|&x| x as f32 / 255.0);
    slice(img, size)
}
>>>>>>> Stashed changes

fn main() {
    /*let trn_size: usize = 1000;
    let tst_size: usize = 20;

    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size as u32)
        .test_set_length(tst_size as u32)
        .label_format_one_hot()
        .finalize();

<<<<<<< Updated upstream
    let mut trn_img_vec: Vec<Vec<f32>> = vec![];

    for img in trn_img.chunks(28 * 28) {
        let img: Vec<f32> = img.into_iter().map(|&x| x as f32 / 255.0).collect();
        trn_img_vec.push(img);
    }

    let mut trn_lbl_vec: Vec<Vec<f32>> = vec![];

    for lbl in trn_lbl.chunks(10) {
        let lbl: Vec<f32> = lbl.into_iter().map(|&x| x as f32).collect();
        trn_lbl_vec.push(lbl);
    }

    let mut model = Model::new(vec![28, 28, 1]);
=======
    let trn_img = img(trn_img, trn_size);
    let trn_lbl = lbl(trn_lbl, trn_size);

    let tst_img = img(tst_img, tst_size);
    let tst_lbl = lbl(tst_lbl, tst_size);

    let mut model = ModelBuilder::new(vec![28, 28, 1]);
>>>>>>> Stashed changes
    model.push_layer(Conv2D::<ReLU>::new(2, (3, 3)));
    model.push_layer(MaxPooling2D::new((2, 2)));
    model.push_layer(Conv2D::<ReLU>::new(4, (4, 4)));
    model.push_layer(MaxPooling2D::new((2, 2)));
    model.push_layer(Flatten::new());
    model.push_layer(Feed::<Sigmoid>::new(20));
    model.push_layer(Feed::<Softmax>::new(10));
    model.train(&trn_img_vec[..], &trn_lbl_vec[..], 20);*/

    let inputs = &[
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = &[vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

<<<<<<< Updated upstream
    let mut model = Model::new(vec![2]);
    model.push_layer(Feed::<Sigmoid>::new(2));
    model.push_layer(Feed::<Sigmoid>::new(1));

    model.compile::<MeanSquaredError>(0.01);

    model.train(inputs, targets, 1000);
=======
    model.train(trn_img, trn_lbl, 10);
    model.test(tst_img);
    for i in 0..tst_lbl.len() {
        println!("{}", tst_lbl[i]);
    }
>>>>>>> Stashed changes
}
