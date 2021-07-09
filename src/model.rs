struct Model {
    series: Series,
}

impl Model {
    pub fn new() -> Self {
        Self {
            series: Series::new(),
        }
    }

    pub fn push(self, layer: Box<dyn Layer>) {}
}
