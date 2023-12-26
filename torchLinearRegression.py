import torch
import matplotlib.pyplot as plt

class LinearModel(torch.nn.Module):
    def __init__(self, lr, epochNum, input_dim):
        super(LinearModel, self).__init__()
        # Adjust the input dimension
        self.linear = torch.nn.Linear(input_dim, 1)
        self.lr = lr
        self.epoch_num = epochNum
        self.criterion = torch.nn.MSELoss()

    def optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    def train(self, x_data, y_data, lossplt=False):
        optimizer = self.optimizer()
        Loss = []
        Loss_x = []
        for epoch in range(self.epoch_num):
            y_pred = self(x_data)
            loss = self.criterion(y_pred, y_data)

            optimizer.zero_grad() # It will be accumulated so we should set zero
            loss.backward()
            optimizer.step() # Update
            Loss.append(loss.item()) #For plot
            Loss_x.append(epoch)

            print(f'Epoch [{epoch + 1}/{self.epoch_num}], Loss: {loss.item()}')

        print('Weights:', self.linear.weight)
        print('Bias:', self.linear.bias)
        if (lossplt):
            plt.plot(Loss_x, Loss)
            plt.xlabel("epoch")
            plt.ylabel("Loss")
            plt.show()



# Example usage:
if __name__ == '__main__':
    # Generate some random data for demonstration
    input_dim = 2  # Number of features
    x_data = torch.rand(100, input_dim)
    y_data = 2 * x_data[:, 0] + 3 * x_data[:, 1] + \
        1  # Example linear relationship

    # Create and train the model
    lr = 0.01
    epochNum = 100
    model = LinearModel(lr, epochNum, input_dim,)
    model.train(x_data, y_data, lossplt=True)

# Here: if __name__ == '__main__': means if the class is imported whihs part won't be excuted
