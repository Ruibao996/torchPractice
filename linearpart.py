import torch
import numpy as np

x_data = [1, 2, 3]
y_data = [2, 4, 6]

w_1 = torch.Tensor([1]) # y_pred = w_1*x^2 + w_2*x + b
w_1.requires_grad = True
w_2 = torch.Tensor([1])
w_2.requires_grad = True


def data_process(x_data):
    x_data_process = []
    for x in x_data:
        x_tmp = []
        x_tmp.append(x**2)
        x_tmp.append(x)
        x_data_process.append(x_tmp)
    return x_data_process

def forward(x):
    return x[0] * w_1 + x[1] * w_2

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print('predict (before training)', 4, forward([16, 4]).item())

for epoch in range(100):
    x_data_process = data_process(x_data)
    for x, y in zip(x_data_process, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrade:', x[1], y, w_1.grad.item(), w_2.grad.item())
        w_1.data = w_1.data - 0.01 * w_1.grad.data
        w_2.data = w_2.data - 0.01 * w_2.grad.data

        w_1.grad.data.zero_()
        w_2.grad.data.zero_()
    
    print("progress:", epoch, l.item())

print("predict (after training)", 4, forward([16, 4]).item())


