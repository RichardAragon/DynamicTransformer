import torch
import torch.nn as nn
import torch.nn.functional as F

def binary_quantize(tensor):
    return torch.sign(tensor)

def ternary_quantize(tensor):
    threshold = 0.05 * tensor.abs().mean()
    tensor = torch.where(tensor > threshold, 1, tensor)
    tensor = torch.where(tensor < -threshold, -1, tensor)
    tensor = torch.where((tensor <= threshold) & (tensor >= -threshold), 0, tensor)
    return tensor

def no_quantize(tensor):
    return tensor

def quantize(tensor, mode):
    if mode == 'binary':
        return binary_quantize(tensor)
    elif mode == 'ternary':
        return ternary_quantize(tensor)
    else:
        return no_quantize(tensor)
