import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# Prepare Dataset

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
    (0.1307,), (0.3081,)
)]) # 0.1307 is the mean of the mnist and 0.3081 is the std of the mnist

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, 
                               download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist', train=False,
                               download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# Softmax Class

class Softmax(torch.nn.Module):
    
    def __init__(self):
        super(Softmax, self).__init__()
        self.l1 = torch.nn.Linear(784, 512) # The matrix size of Mnist is 28*28
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784) # -1 means obtain mini_batch automatically
        x = F.relu(self.l1(x)) # We use a more popular activation function relu than sigmoid
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x) # The last output shouldn't be activated
    
model = Softmax()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)

# Train and Test part

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # Get one batch data and label
        inputs, target = data
        optimizer.zero_grad()
        # Get output
        outputs = model(inputs)
        # Get loss and backward
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Every 300 times print once
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, prediction = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
    print('Accuracy on test dataset: %d %%' % (100*correct/total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

