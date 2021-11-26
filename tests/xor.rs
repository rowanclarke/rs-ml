use ml::{
    activation::{sigmoid::Sigmoid, Activation},
    layer::{feed::Feed, Layer},
    loss::{mse::MeanSquaredError, Loss, LossBuilder},
    matrix::Column,
    model::{Model, ModelBuilder},
};

#[test]
fn create_model() {
    let inputs = [
        Column::new(vec![0.0, 0.0]),
        Column::new(vec![0.0, 1.0]),
        Column::new(vec![1.0, 0.0]),
        Column::new(vec![1.0, 1.0]),
    ];

    let targets = [
        Column::new(vec![0.0]),
        Column::new(vec![1.0]),
        Column::new(vec![1.0]),
        Column::new(vec![0.0]),
    ];

    let mut model = ModelBuilder::new("xor", vec![2]);
    model.push_layer(Feed::<Sigmoid>::new(2));
    model.push_layer(Feed::<Sigmoid>::new(1));

    let mut model = model.compile::<MeanSquaredError>(0.05);

    model.train(Box::new(inputs.clone()), Box::new(targets.clone()), 10_000);

    model.validate(Box::new(inputs.clone()), Box::new(targets.clone()));
    model.test(Box::new(inputs.clone()));
}
