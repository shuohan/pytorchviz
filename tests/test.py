#!/usr/bin/env python

import torch
from pytorchviz import make_dot


class MLP(torch.nn.Sequential):
    def __init__(self):
        super().__init__()
        self.w0 = torch.nn.Linear(8, 16)
        self.tanh = torch.nn.Tanh()
        self.w1 = torch.nn.Linear(16, 1)


class DoubleInputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 8, 3, padding=1)
        self.norm = torch.nn.BatchNorm2d(8)
        self.activ = torch.nn.ReLU()
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activ(out)
        out = out + x1
        return out


class DoublePropModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
    def forward(self, x):
        y = self.mlp(x).mean()
        grad, = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)
        y = grad.pow(2).mean() + y
        return y


def test_mlp():
    x = torch.randn(1, 8)
    mlp = MLP()
    dot = make_dot(x, mlp)
    dot.render('mlp')


def test_double_backprop_model():
    x = torch.randn(1, 8)
    dp = DoublePropModel()
    dot = make_dot(x, dp)
    dot.render('double_backprop')


def test_lstm():
    x = torch.randn(1, 128)
    lstm_cell = torch.nn.LSTMCell(128, 128)
    dot = make_dot(x, lstm_cell)
    dot.render('lstm')


def test_double_input_model():
    di = DoubleInputModel()
    x1 = torch.randn(1, 8, 16, 16)
    x2 = torch.randn(1, 8, 16, 16)
    dot = make_dot((x1, x2), di)
    dot.render('double_input', format='png')


if __name__ == '__main__':
    test_mlp()
    test_double_backprop_model()
    test_double_input_model()
    test_lstm()
