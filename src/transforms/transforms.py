import torch
import torch.nn as nn

class Noise(nn.Module):
    """Adds a small perturbation to the image colors"""
    def __init__(self, magnitude=0.25, p=0.5):
        super().__init__()
        self.magnitude = magnitude
        self.p = p

    def __call__(self, image):
        if self.p < torch.rand(1):
            perturbation = torch.randn_like(image) * self.magnitude
            return torch.clamp(image + (image * perturbation), min=0, max=1)
        else:
            return image