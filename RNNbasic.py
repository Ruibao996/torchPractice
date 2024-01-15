# In this part I will build a module using RNN to make hello transformed to oholo and I will
# use a powerful and popular way called embedding to make it more efficient.

import torch
import torch.optim as optim

# Prepare Dataset
input_size = 4
hidden_size = 8
batch_size = 1
num_layers = 2
num_classes = 4
seq_len = 5
embedding_size = 10

idx2char = ['e', 'h', 'l', 'o']
x_data = [[1, 0, 2, 2, 3]] # hello (batch, seq_len)
y_data = [3, 1, 2, 3, 2] # ohlol (batch*seq_len)


inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        return x.view(-1, num_classes)
    
net = RNN()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.05)

if __name__ == '__main__':
    for epoch in range(15):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, idx = outputs.max(dim=1)
        idx = idx.data.numpy()
        print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
        print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))
