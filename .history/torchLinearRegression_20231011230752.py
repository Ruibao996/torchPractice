# import torch

# class LinearModel(torch.nn.Module):
#     def __init__(self, lr, epochNum):
#         super(LinearModel, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#         self.lr = lr
#         self.epoch_num = epochNum
#         self.criterion = torch.nn.MSELoss()  # Use the default mean squared error loss

#     def optimizer(self):
#         return torch.optim.SGD(self.parameters(), lr=self.lr)

#     def forward(self, x):
#         y_pred = self.linear(x)
#         return y_pred

#     def train(self, x_data, y_data):
#         optimizer = self.optimizer()
#         for epoch in range(self.epoch_num):
#             y_pred = self(x_data)
#             loss = self.criterion(y_pred, y_data)

#             # You should zero the gradients of the model's parameters, not the optimizer
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             print(f'Epoch [{epoch + 1}/{self.epoch_num}], Loss: {loss.item()}')

#         print('w = ', self.linear.weight.item())
#         print('b = ', self.linear.bias.item())

# # Example usage:
# if __name__ == '__main__':
#     # Generate some random data for demonstration
#     x_data = torch.rand(100, 1)
#     y_data = 2 * x_data + 1

#     # Create and train the model
#     lr = 0.01
#     epochNum = 100
#     model = LinearModel(lr, epochNum)
#     model.train(x_data, y_data)

    
import torch

class LinearModel(torch.nn.Module):
    def __init__(self, lr, epochNum, input_dim):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)  # Adjust the input dimension
        self.lr = lr
        self.epoch_num = epochNum
        self.criterion = torch.nn.MSELoss()

    def optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    def train(self, x_data, y_data):
        optimizer = self.optimizer()
        for epoch in range(self.epoch_num):
            y_pred = self(x_data)
            loss = self.criterion(y_pred, y_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch + 1}/{self.epoch_num}], Loss: {loss.item()}')

        print('Weights:', self.linear.weight)
        print('Bias:', self.linear.bias)

# Example usage:
if __name__ == '__main__':
    # Generate some random data for demonstration
    input_dim = 2  # Number of features
    x_data = torch.rand(100, input_dim)
    y_data = 2 * x_data[:, 0] + 3 * x_data[:, 1] + 1  # Example linear relationship

    # Create and train the model
    lr = 0.01
    epochNum = 100
    model = LinearModel(lr, epochNum, input_dim)
    model.train(x_data, y_data)
