use super::super::{
    activation::Activation,
    matrix::{Column, Jacobean, Matrix},
};
use super::{Cost, CostObject, Layer, LayerBuilder};
use ndarray::{Array2, Array3, Array4, Ix2};
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

impl<A: Activation> LayerBuilder for Conv2D<A> {
    fn build(&self, before: Vec<usize>, other: CostObject) -> Box<dyn Layer> {
        let mut rng = rand::thread_rng();
        let after = vec![
            before[0] - self.kernel_size.0 + 1,
            before[1] - self.kernel_size.1 + 1,
            self.filters,
        ];
        Box::new(Conv2DLayer::<A> {
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
            other,
            phantom: PhantomData,
        })
    }

    fn after(&self, before: Vec<usize>) -> Vec<usize> {
        vec![
            before[0] - self.kernel_size.0 + 1,
            before[1] - self.kernel_size.1 + 1,
            self.filters,
        ]
    }
}

pub struct Conv2DLayer<A: Activation> {
    filter: Array4<f32>,
    input: Array3<f32>,
    sum: Array3<f32>,
    output: Array3<f32>,
    before: Vec<usize>,
    after: Vec<usize>,
    other: CostObject,
    phantom: PhantomData<A>,
}

impl<A: Activation> Cost for Conv2DLayer<A> {
    fn cost(&self, given: Jacobean) -> Jacobean {
        let sum = self.sum.clone().into_raw_vec();
        let ds_x = self.ds_x();
        let da_s = A::deactivate(Array2::from_shape_vec((sum.len(), 1), sum).unwrap());
        self.other.cost(da_s.dot(&ds_x))
    }
}

impl<A: Activation> Layer for Conv2DLayer<A> {
    fn before(&self) -> Vec<usize> {
        self.before.clone()
    }

    fn after(&self) -> Vec<usize> {
        self.after.clone()
    }

    fn train(&mut self, input: Column, target: Column, lr: f32) {
        self.forward(input);
        let output = self.output.clone().into_raw_vec();
        let output = Array2::from_shape_vec((output.len(), 1), output).unwrap();
        match &mut self.other {
            CostObject::Layer(layer) => layer.train(output, target, lr),
            CostObject::Loss(loss) => loss.train(output, target),
        };
        self.backward(lr);
    }

    fn test(&mut self, input: Column) {
        self.forward(input);
        let output = self.output.clone().into_raw_vec();
        let output = Array2::from_shape_vec((output.len(), 1), output).unwrap();
        match &mut self.other {
            CostObject::Layer(layer) => layer.test(output),
            CostObject::Loss(_) => println!("{}", self.output),
        };
    }

    fn forward(&mut self, input: Column) {
        self.input = Array3::from_shape_vec(
            (self.before[0], self.before[1], self.before[2]),
            input.into_raw_vec(),
        )
        .unwrap();
        Self::convolution(&self.input, &self.filter, &mut self.sum);
        let sum = self.sum.clone().into_raw_vec();
        self.output = Array3::from_shape_vec(
            (self.after[0], self.after[1], self.after[2]),
            A::activate(Array2::from_shape_vec((sum.len(), 1), sum).unwrap()).into_raw_vec(),
        )
        .unwrap();
    }

    fn backward(&mut self, lr: f32) {
        let sum = self.sum.clone().into_raw_vec();
        let mut delf = Array4::<f32>::zeros(self.filter.raw_dim());
        let dc_a = Array2::ones((1, sum.len())).dot(&self.other.cost(A::deactivate(
            Array2::from_shape_vec((sum.len(), 1), sum).unwrap(),
        )));
        let dc_a = Array3::from_shape_vec(
            (self.after[0], self.after[1], self.after[2]),
            dc_a.into_raw_vec(),
        )
        .unwrap();
        Self::back_convolution(&self.input, &dc_a, &mut delf);
        self.filter = &self.filter - &(lr * delf);
    }
}

impl<A: Activation> Conv2DLayer<A> {
    pub fn ds_x(&self) -> Array2<f32> {
        let mut ds_x = Array2::<f32>::zeros((self.output.len(), self.input.len()));
        let li = self.input.shape();
        let lo = self.output.shape();
        let lf = self.filter.shape();
        for i in 0..lo[0] {
            for j in 0..lo[1] {
                for k in 0..lo[2] {
                    for x in 0..lf[0] {
                        for y in 0..lf[1] {
                            for z in 0..lf[2] {
                                ds_x[[
                                    lo[2] * (lo[1] * i + j) + k,
                                    lf[2] * (li[1] * (x + i) + y + j) + z,
                                ]] = self.filter[[x, y, z, k]];
                            }
                        }
                    }
                }
            }
        }
        ds_x
    }

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
                for k in 0..output.shape()[2] {
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
