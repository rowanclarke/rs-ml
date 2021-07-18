use super::super::activation::Activation;
use super::{Layer, Template};
use ndarray::{Array1, Array2, Array3, Array4};
use rand::prelude::*;
use std::marker::PhantomData;

pub struct Conv2D<A: Activation> {
    filters: usize,
    kernel_size: (usize, usize),
    phantom: PhantomData<A>,
}

impl<A: Activation> Conv2D<A> {
    pub fn new(filters: usize, kernel_size: (usize, usize)) -> Self {
        Self {
            filters,
            kernel_size,
            phantom: PhantomData,
        }
    }
}

impl<A: Activation> Template<Conv2DLayer<A>> for Conv2D<A> {
    fn into(self, before: Vec<usize>) -> Conv2DLayer<A> {
        let mut rng = rand::thread_rng();
        let after = vec![
            before[0] - self.kernel_size.0 + 1,
            before[1] - self.kernel_size.1 + 1,
            self.filters,
        ];
        Conv2DLayer::<A> {
            filter: Array4::<f32>::zeros((
                self.kernel_size.0,
                self.kernel_size.1,
                before[2],
                self.filters,
            ))
            .map(|_| rng.gen::<f32>()),
            input: Array3::<f32>::zeros((before[0], before[1], before[2])),
            sum: Array3::<f32>::zeros((after[0], after[1], after[2])),
            output: Array3::<f32>::zeros((after[0], after[1], after[2])),
            before,
            after,
            phantom: PhantomData,
        }
    }
}

pub struct Conv2DLayer<A: Activation> {
    filter: Array4<f32>,
    input: Array3<f32>,
    sum: Array3<f32>,
    output: Array3<f32>,
    before: Vec<usize>,
    after: Vec<usize>,
    phantom: PhantomData<A>,
}

impl<A: Activation> Layer for Conv2DLayer<A> {
    fn before(&self) -> Vec<usize> {
        self.before.clone()
    }

    fn after(&self) -> Vec<usize> {
        self.after.clone()
    }

    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        self.input =
            Array3::from_shape_vec((self.before[0], self.before[1], self.before[2]), input)
                .unwrap();
        Self::convolution(&self.input, &self.filter, &mut self.sum);
        self.output = Array3::from_shape_vec(
            (self.after[0], self.after[1], self.after[2]),
            A::activate(self.sum.clone().into_raw_vec()),
        )
        .unwrap();
        self.output.clone().into_raw_vec()
    }

    fn backward(&mut self, target: Vec<f32>, lr: f32) -> Vec<f32> {
        let mut delf = Array4::<f32>::zeros(self.filter.raw_dim());
        let mut deli = Array3::<f32>::zeros(self.input.raw_dim());
        let del = Array3::from_shape_vec(
            (self.after[0], self.after[1], self.after[2]),
            A::deactivate(self.sum.clone().into_raw_vec()),
        )
        .unwrap();
        let target =
            Array3::from_shape_vec((self.after[0], self.after[1], self.after[2]), target).unwrap();
        let dele = (self.output.clone() - target) * del;

        Self::back_convolution(&self.input, &dele, &mut delf);
        Self::full_convolution_rot(&dele, &self.filter, &mut deli);

        self.filter = self.filter.clone() - lr * delf;
        (self.input.clone() - lr * deli).into_raw_vec()
    }
}

impl<A: Activation> Conv2DLayer<A> {
    pub fn convolution(input: &Array3<f32>, filter: &Array4<f32>, output: &mut Array3<f32>) {
        for i in 0..output.shape()[0] {
            for j in 0..output.shape()[1] {
                for k in 0..output.shape()[2] {
                    let mut sum = 0.0;
                    for x in 0..filter.shape()[0] {
                        for y in 0..filter.shape()[1] {
                            for z in 0..filter.shape()[2] {
                                sum += input[[i + x, j + y, z]] * filter[[x, y, z, k]];
                            }
                        }
                    }
                    output[[i, j, k]] = sum;
                }
            }
        }
    }

    pub fn back_convolution(input: &Array3<f32>, output: &Array3<f32>, filter: &mut Array4<f32>) {
        for i in 0..filter.shape()[0] {
            for j in 0..filter.shape()[1] {
                for k in 0..filter.shape()[2] {
                    for l in 0..filter.shape()[3] {
                        let mut sum = 0.0;
                        for x in 0..output.shape()[0] {
                            for y in 0..output.shape()[1] {
                                sum += input[[i + x, j + y, k]] * output[[x, y, l]];
                            }
                        }
                        filter[[i, j, k, l]] = sum;
                    }
                }
            }
        }
    }

    pub fn full_convolution_rot(
        input: &Array3<f32>,
        filter: &Array4<f32>,
        output: &mut Array3<f32>,
    ) {
        for i in 0..output.shape()[0] {
            for j in 0..output.shape()[1] {
                for k in 0..output.shape()[1] {
                    let mut sum = 0.0;
                    for x in 0..filter.shape()[0] {
                        for y in 0..filter.shape()[1] {
                            for z in 0..filter.shape()[2] {
                                let xi = i as i32 - x as i32;
                                let yj = j as i32 - y as i32;
                                if xi >= 0
                                    && yj >= 0
                                    && xi < input.shape()[0] as i32
                                    && yj < input.shape()[1] as i32
                                {
                                    sum +=
                                        input[[xi as usize, yj as usize, z]] * filter[[x, y, z, k]];
                                }
                            }
                        }
                    }
                    output[[i, j, k]] = sum;
                }
            }
        }
    }
}

pub struct MaxPooling2D {
    pool_size: (usize, usize),
}

impl MaxPooling2D {
    pub fn new(pool_size: (usize, usize)) -> Self {
        Self { pool_size }
    }
}

impl Template<MaxPooling2DLayer> for MaxPooling2D {
    fn into(self, before: Vec<usize>) -> MaxPooling2DLayer {
        let after = vec![
            before[0] / self.pool_size.0,
            before[1] / self.pool_size.1,
            before[2],
        ];
        MaxPooling2DLayer {
            pool_size: self.pool_size,
            input: Array3::<f32>::zeros((before[0], before[1], before[2])),
            output: Array3::<f32>::zeros((before[0], before[1], before[2])),
            before,
            after,
        }
    }
}

pub struct MaxPooling2DLayer {
    pool_size: (usize, usize),
    input: Array3<f32>,
    output: Array3<f32>,
    before: Vec<usize>,
    after: Vec<usize>,
}

impl Layer for MaxPooling2DLayer {
    fn before(&self) -> Vec<usize> {
        self.before.clone()
    }

    fn after(&self) -> Vec<usize> {
        self.after.clone()
    }

    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        self.input =
            Array3::from_shape_vec((self.before[0], self.before[1], self.before[2]), input)
                .unwrap();
        for i in 0..self.after[0] {
            for j in 0..self.after[1] {
                for k in 0..self.after[2] {
                    let mut max: f32 = 0.0;
                    for x in 0..self.pool_size.0 {
                        for y in 0..self.pool_size.1 {
                            max = max.max(
                                self.input[[self.pool_size.0 * i + x, self.pool_size.1 * j + y, k]],
                            )
                        }
                    }
                    self.output[[i, j, k]] = max;
                }
            }
        }
        self.output.clone().into_raw_vec()
    }

    fn backward(&mut self, target: Vec<f32>, lr: f32) -> Vec<f32> {
        let target =
            Array3::from_shape_vec((self.after[0], self.after[1], self.after[2]), target).unwrap();
        for i in 0..self.before[0] {
            for j in 0..self.before[1] {
                for k in 0..self.before[2] {
                    if self.input[[i, j, k]]
                        == self.output[[i / self.pool_size.0, j / self.pool_size.1, k]]
                    {
                        self.input[[i, j, k]] =
                            target[[i / self.pool_size.0, j / self.pool_size.1, k]];
                    }
                }
            }
        }
        self.input.clone().into_raw_vec()
    }
}
