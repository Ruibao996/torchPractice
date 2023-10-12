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
    
    def train(self, x_data, y_data, predPlt=False, x_rand=[0, 10, 200]):
        optimizer = self.optimizer()
        for epoch in range(self.epoch_num):
            y_pred = self(x_data)
            loss = self.criterion(torch.sigmoid(y_pred), y_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f'Epoch[{epoch + 1}/{self.epoch_num}], Loss: {loss.item()}')

        if self.input_dim == 1 and predPlt == True:
            x = torch.linspace(x_rand[0], x_rand[1], x_rand[2])
            x_t = torch.Tensor(x).view((x_rand[2], 1))
            y_t = self(x_t)
            y = y_t.data.numpy()
            plt.plot(x, y)
            plt.grid()
            plt.show()

# Example usage:
if __name__ == '__main__':
    # Generate some random data for demonstration
    input_dim = 1  # Number of features
    x_data = torch.rand(100, input_dim)
    y_data = torch.rand(100, 1)

    # Create and train the model
    lr = 0.01
    epochNum = 100
    model = LogisticRegressionModel(lr, epochNum, input_dim)
    model.train(x_data, y_data, predPlt=True)
    print(x_data)
    print(y_data)
