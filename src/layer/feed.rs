extern crate ndarray;
extern crate rand;

use super::super::activation::Activation;
use super::{Fixed, Layer};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use std::marker::PhantomData;

pub struct Feed<A: Activation> {
    weights: Array2<f32>,
    bias: Array1<f32>,
    before: usize,
    after: usize,
    input: Vec<f32>,
    sum: Vec<f32>,
    output: Vec<f32>,
    phantom: PhantomData<A>,
}

impl<A: Activation> Fixed for Feed<A> {
    fn new(before: usize, after: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            weights: Array2::<f32>::zeros((before, after)).map(|_| rng.gen::<f32>()),
            bias: Array1::<f32>::zeros(after).map(|_| rng.gen::<f32>()),
            before,
            after,
            input: Vec::new(),
            sum: Vec::new(),
            output: Vec::new(),
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
        self.input = input;
        self.sum = vec![0.0; self.after];
        for i in 0..self.after {
            for j in 0..self.before {
                self.sum[i] += self.bias[i] + self.weights[[j, i]] * self.input[j];
            }
        }
        self.output = A::activate(self.sum.clone());
        self.output.clone()
    }

    fn backward(&mut self, target: Vec<f32>) -> Vec<f32> {
        let mut result = self.input.clone();
        let del = A::deactivate(self.sum.clone());
        for j in 0..self.before {
            for i in 0..self.after {
                let buffer = 0.5 * 2.0 * (self.output[i] - target[i]) * del[i];
                result[j] -= buffer * self.weights[[j, i]] * self.input[j];
                self.weights[[j, i]] -= buffer * self.input[j];
            }
        }
        result
    }
}
