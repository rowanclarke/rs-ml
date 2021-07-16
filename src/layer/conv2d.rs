use super::super::activation::Activation;
use super::{Fixed, Layer, Template};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use std::marker::PhantomData;

pub struct Conv2D {
    filter: Array2<f32>,
    input: Array2<f32>,
    output: Array2<f32>,
    before: Vec<usize>,
    after: Vec<usize>,
}

pub struct TempConv2D {
    pub filters: usize,
    pub kernel_size: (usize, usize),
}

impl Template<Conv2D> for TempConv2D {
    fn into(self, before: Vec<usize>) -> Conv2D {
        let mut rng = rand::thread_rng();
        let after = vec![
            before[0] - self.kernel_size.0 + 1,
            before[1] - self.kernel_size.1 + 1,
        ];
        Conv2D {
            filter: Array2::<f32>::zeros((self.kernel_size.0, self.kernel_size.1))
                .map(|_| rng.gen::<f32>()),
            input: Array2::<f32>::zeros((before[0], before[1])).map(|_| rng.gen::<f32>()),
            output: Array2::<f32>::zeros((after[0], after[1])).map(|_| rng.gen::<f32>()),
            before,
            after,
        }
    }
}

impl Layer for Conv2D {
    fn before(&self) -> Vec<usize> {
        self.before.clone()
    }

    fn after(&self) -> Vec<usize> {
        self.after.clone()
    }

    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        self.input = Array2::from_shape_vec((self.before[0], self.before[1]), input).unwrap();
        Self::convolution(&self.input, &self.filter, &mut self.output);
        self.output.clone().into_raw_vec()
    }

    fn backward(&mut self, target: Vec<f32>, lr: f32) -> Vec<f32> {
        let mut delf = Array2::<f32>::zeros(self.filter.raw_dim());
        let mut deli = Array2::<f32>::zeros(self.input.raw_dim());
        let target = Array2::from_shape_vec((self.after[0], self.after[1]), target).unwrap();
        let dele = 2.0 * (self.output.clone() - target);

        Self::convolution(&self.input, &dele, &mut delf);
        Self::full_convolution_rot(&dele, &self.filter, &mut deli);

        self.filter = self.filter.clone() - lr * delf;
        (self.input.clone() - lr * deli).into_raw_vec()
    }
}

impl Conv2D {
    pub fn convolution(input: &Array2<f32>, filter: &Array2<f32>, output: &mut Array2<f32>) {
        for i in 0..output.shape()[0] {
            for j in 0..output.shape()[1] {
                let mut sum = 0.0;
                for x in 0..filter.shape()[0] {
                    for y in 0..filter.shape()[1] {
                        sum += input[[i + x, j + y]] * filter[[x, y]];
                    }
                }
                output[[i, j]] = sum;
            }
        }
    }

    pub fn full_convolution_rot(
        input: &Array2<f32>,
        filter: &Array2<f32>,
        output: &mut Array2<f32>,
    ) {
        for i in 0..output.shape()[0] {
            for j in 0..output.shape()[1] {
                let mut sum = 0.0;
                for x in 0..filter.shape()[0] {
                    for y in 0..filter.shape()[1] {
                        let xi = i as i32 - x as i32;
                        let yj = j as i32 - y as i32;
                        if xi >= 0
                            && yj >= 0
                            && xi < input.shape()[0] as i32
                            && yj < input.shape()[1] as i32
                        {
                            sum += input[[xi as usize, yj as usize]] * filter[[x, y]];
                        }
                    }
                }
                output[[i, j]] = sum;
            }
        }
    }
}
