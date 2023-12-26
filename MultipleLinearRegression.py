import torch
import numpy as np 

class MultipleLinearModel(torch.nn.Module):
    # In this case we deal with the data with 8 dimension
    def __init__(self):
        super(MultipleLinearModel, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.criterion = torch.nn.BCELoss(reduction='sum')

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x
    
    def optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)
    
    def train(self, x_data, y_data):
        optimizer = self.optimizer()
        epoch_num = 100
        # Forward
        for epoch in range(epoch_num):
            # Forward
            y_pred = self(x_data)
            loss = self.criterion(y_pred, y_data)
            print(epoch, loss.item())

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Update
            optimizer.step()

    

# Example Usage

if __name__ == '__main__':
    xy = np.loadtxt('/Users/liruifeng/Desktop/Something for Fun/torchPractice/diabetes.csv.gz', delimiter=',', dtype=np.float32)
    x_data = torch.from_numpy(xy[:, :-1])
    y_data = torch.from_numpy(xy[:, [-1]]) # To make y_data in Matrix
    model = MultipleLinearModel()
    model.train(x_data, y_data)