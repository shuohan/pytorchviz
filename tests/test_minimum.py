#!/usr/bin/env python

import torch
from pytorchviz import make_dot
# from torchviz import make_dot


x = torch.randn(1, 2, 3)

class Minimum(torch.nn.Module):
    def forward(self, x):
        return x + 2

x.requires_grad = True
y = Minimum()(x)
print(y.grad_fn.next_functions)
dot = make_dot(x, Minimum())
# x.requires_grad = True
# model = Minimum()
# params = dict(model.named_parameters())
# params['x'] = x
# dot = make_dot(model(x), params)
dot.render('minimum')
