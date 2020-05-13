#!/usr/bin/env python

import torch
from torch import nn
from torchviz import make_dot


def make_mlp_and_input():
    model = nn.Sequential()
    model.add_module('W0', nn.Linear(8, 16))
    model.add_module('tanh', nn.Tanh())
    model.add_module('W1', nn.Linear(16, 1))
    x = torch.randn(1, 8)
    return model, x


def make_double_backprop(x, model):
    y = model(x).mean()
    grad, = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)
    return grad.pow(2).mean() + y


def test_mlp_make_dot():
    model, x = make_mlp_and_input()
    y = model(x)
    dot = make_dot(y.mean(), params=dict(model.named_parameters()))
    dot.render('mlp')


def test_double_backprop_make_dot():
    model, x = make_mlp_and_input()
    x.requires_grad = True
    y = make_double_backprop(x, model)
    params = dict(model.named_parameters())
    params['x'] = x
    dot = make_dot(y, params=params)
    dot.render('double_backprop')


def test_lstm_make_dot():
    lstm_cell = nn.LSTMCell(128, 128)
    x = torch.randn(1, 128)
    dot = make_dot(lstm_cell(x), params=dict(lstm_cell.named_parameters()))
    dot.render('lstm')


if __name__ == '__main__':
    test_mlp_make_dot()
    test_double_backprop_make_dot()
    test_lstm_make_dot()
