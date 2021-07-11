extern crate ndarray;

use super::super::activation::Activation;
use super::{Fixed, Layer};
use ndarray::Array2;
use std::marker::PhantomData;

pub struct Feed<A: Activation> {
    weights: Array2<f32>,
    bias: Vec<f32>,
    before: usize,
    after: usize,
    phantom: PhantomData<A>,
}

impl<A: Activation> Fixed for Feed<A> {
    fn new(before: usize, after: usize) -> Self {
        Self {
            weights: Array2::zeros((after, before)),
            bias: vec![0.0; after],
            before,
            after,
            phantom: PhantomData,
        }
    }
}

impl<A: Activation> Layer for Feed<A> {
    fn before(&self) -> usize {
        self.before
    }

    fn after(&self) -> usize {
        self.after
    }

    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut sum = vec![0.0; self.after];
        for i in 0..self.after {
            for j in 0..self.before {
                sum[i] += self.bias[i] + self.weights[[i, j]] * input[j];
            }
        }
        A::activate(sum)
    }

    fn backward(&mut self) {}
}
