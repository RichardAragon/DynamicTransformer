class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuantizedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mode = 'full'  # Default mode

    def forward(self, x):
        weight = quantize(self.linear.weight, self.mode)
        bias = quantize(self.linear.bias, self.mode)
        return F.linear(x, weight, bias)

    def set_mode(self, mode):
        self.mode = mode
