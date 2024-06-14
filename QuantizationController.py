class QuantizationController:
    def __init__(self, model):
        self.model = model

    def set_mode(self, mode):
        for layer in self.model.modules():
            if isinstance(layer, QuantizedLinear):
                layer.set_mode(mode)

    def adjust_mode(self, performance_metric):
        if performance_metric < 0.5:
            self.set_mode('full')
        elif performance_metric < 0.8:
            self.set_mode('ternary')
        else:
            self.set_mode('binary')
