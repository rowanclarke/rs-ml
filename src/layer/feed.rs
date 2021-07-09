extern crate ndarray;

use super::super::activation::Activation;
use super::Layer;
use ndarray::Array2;
use std::marker::PhantomData;

pub struct Feed<A: Activation> {
    weights: Array2<f32>,
    bias: Vec<f32>,
    phantom: PhantomData<A>,
}

impl<A: Activation> Feed<A> {
    pub fn new(size: usize, prev: usize) -> Self {
        Self {
            weights: Array2::zeros((size, prev)),
            bias: vec![0.0; size],
            phantom: PhantomData,
        }
    }
}

impl<A: Activation + 'static> Layer for Feed<A> {
    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let shape = self.weights.shape();
        let mut sum = vec![0.0; shape[0]];
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                sum[i] += self.bias[i] + self.weights[[j, i]] * input[j];
            }
        }
        A::activate(sum)
    }

    fn backward(&mut self) {}
}
