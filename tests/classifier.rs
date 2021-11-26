use ml::{
  activation::{sigmoid::Sigmoid, softmax::Softmax, Activation},
  layer::{feed::Feed, Layer},
  loss::{mse::MeanSquaredError, Loss, LossBuilder},
  matrix::Column,
  model::{Model, ModelBuilder},
};
use rand::prelude::*;

fn generate_inputs(train: usize, input: usize) -> Vec<Column> {
  let mut inputs: Vec<Column> = vec![];
  let mut rng = rand::thread_rng();
  for _ in [0..train] {
    let mut column: Vec<f32> = vec![];
    for _ in [0..input] {
      column.push((rng.next_u32() % 2) as f32)
    }
    inputs.push(Column::new(column));
  }
  inputs
}

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

  println!("{:#?}", generate_inputs(1, 1));

  /*
  let mut model = ModelBuilder::new("xor", vec![2]);
  model.push_layer(Feed::<Sigmoid>::new(2));
  model.push_layer(Feed::<Sigmoid>::new(1));

  let mut model = model.compile::<MeanSquaredError>(0.05);

  model.train(Box::new(inputs.clone()), Box::new(targets.clone()), 10_000);

  model.validate(Box::new(inputs.clone()), Box::new(targets.clone()));
  model.test(Box::new(inputs.clone()));*/
}
