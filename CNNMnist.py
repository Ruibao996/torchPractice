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

# CNN Class

class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)

        x = self.fc(x)

        return x
    
model = CNN()
# Add GPU, for mac or AMD GPUs we can use mps to accelerate by GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)

# Train and Test part
# GPU should be also impleted in tain and test

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # Get one batch data and label
        inputs, target = data

        # GPU
        inputs, target = inputs.to(device), target.to(device)

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

            # GPU
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, prediction = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
    print('Accuracy on test dataset: %.3f %%' % (100*correct/total))


if __name__ == '__main__':
    for epoch in range(20):
        train(epoch)
        test()
