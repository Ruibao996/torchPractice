import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, lr, epoch_num, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.input_dim = input_dim
        self.lr = lr
        self.epoch_num = epoch_num
        self.criterion = torch.nn.BCELoss(size_average=False)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
    
    def optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
    
    def train(self, x_data, y_data, predPlt=False):
        optimizer = self.optimizer()
        for epoch in range(self.epoch_num):
            y_pred = self(x_data)
            loss = self.criterion(torch.sigmoid(y_pred), y_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f'Epoch[{epoch + 1}/{self.epoch_num}], Loss: {loss.item()}')

        if self.input_dim == 1 and predPlt == True:
            x = x_data.data.numpy()
            y = y_pred.data.numpy()
            plt.plot(x, y)
            plt.grid()
            plt.show()

        if self.input_dim == 2 and predPlt == True:
            x = x_data.data.numpy()
            y = y_pred.data.numpy()
            plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='RdYlGn')
            plt.grid()
            plt.show()


# # if __name__ == '__main__':
# #     # Generate some random data for demonstration
# #     input_dim = 2  # Number of features
# #     x_data = torch.rand(100, input_dim)
# #     y_data = torch.zeros(100)
# #     for i in range(100):
# #         if x_data[i, 0] + x_data[i, 1] > 1:
# #             y_data[i] = 1
# #         else:
# #             y_data[i] = 0

# #     # Create and train the model
# #     lr = 0.01
# #     epochNum = 100
# #     model = LogisticRegressionModel(lr, epochNum, input_dim)
# #     model.train(x_data, y_data, predPlt=True)

# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# class LogisticRegressionModel(torch.nn.Module):
#     def __init__(self, lr, epoch_num, input_dim):
#         super(LogisticRegressionModel, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, 1)
#         self.input_dim = input_dim
#         self.lr = lr
#         self.epoch_num = epoch_num
#         self.criterion = torch.nn.BCEWithLogitsLoss(size_average=False)  # Use BCEWithLogitsLoss for stability

#     def forward(self, x):
#         y_pred = self.linear(x)
#         return y_pred
    
#     def optimizer(self):
#         return torch.optim.SGD(self.parameters(), lr=self.lr)
    
#     def train(self, x_data, y_data, predPlt=False):
#         optimizer = self.optimizer()
#         for epoch in range(self.epoch_num):
#             y_pred = self(x_data)
#             loss = self.criterion(y_pred, y_data)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             print(f'Epoch[{epoch + 1}/{self.epoch_num}], Loss: {loss.item()}')

#         if self.input_dim == 1 and predPlt:
#             x = x_data.data.numpy()
#             y = y_pred.data.numpy()
#             plt.plot(x, y, label='Fitted Line')
#             plt.scatter(x, y_data, label='Data Points')
#             plt.xlabel('X')
#             plt.ylabel('Predicted Probability')
#             plt.legend()
#             plt.grid()
#             plt.show()

# Example usage:
if __name__ == '__main__':
    # Generate some random data for demonstration
    input_dim = 1  # Number of features
    x_data = torch.Tensor([[1.0], [2.0], [3.0]])
    y_data = torch.Tensor([[0], [0], [1]])

    # Create and train the model
    lr = 0.01
    epochNum = 100
    model = LogisticRegressionModel(lr, epochNum, input_dim)
    model.train(x_data, y_data, predPlt=True)
