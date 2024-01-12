import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class titanicSet(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        xy = np.delete(xy, 0, axis=0)
        self.len = xy.shape[0]
        self.labels = xy[0]
        xy = np.delete(xy, [0, 3, -2], axis=1)
        self.x_data = torch.from_numpy(xy[:, 1:])
        self.y_data = torch.from_numpy(xy[:, [0]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
dataset = titanicSet('/Users/liruifeng/Desktop/Something for Fun/torchPractice/titanicTrain.csv')
# datadataset = titanicSet('/Users/liruifeng/Desktop/Something for Fun/torchPractice/titanicTest.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=4)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(6, 4)
        self.linear2 = torch.nn.Linear(4, 2)
        self.linear3 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x      

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = Model().to(device)

if __name__ == '__main__':

    criterion = torch.nn.BCELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            # 1. prepare data
            inputs, labels = data

            # 2. Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            # 3. Backward
            optimizer.zero_grad()
            loss.backward()

            #  4. Update
            optimizer.step()

# 在深度学习中，"momentum"是一种用于加速梯度下降算法的技术，可以帮助模型更快地收敛，并且可以减少模型陷入局部最优解的风险。
# 在这段代码中，momentum被设置为0.5。这意味着在每次更新模型参数时，除了当前梯度值，还会考虑到前一次梯度的方向。这样可以使得模型在梯度下降过程中具有一定的“惯性”，即使梯度发生变化，模型也会继续沿着原来的方向前进，从而加快收敛速度并减少震荡。
