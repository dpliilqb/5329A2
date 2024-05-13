import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, kernel_size=4, padding=1, stride=2)
        self.conv2 = nn.Conv2d(9, 27, kernel_size=4, padding=1, stride=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(27*64*64, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 18)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(-1, 27*64*64)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def predict(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def loss_function(self, x, y):
        return self.loss(x, y)

    def accuracy(self, x, y):
        pred = self.predict(x)
        return self.accuracy(pred, y)

    def evaluate(self, x, y):
        pred = self.predict(x)
        return self.accuracy(pred, y)

    def train_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        pred = self.forward(x)
        loss = self.loss(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss, pred

    def test_step(self, x, y):
        pred = self.predict(x)
        loss = self.loss(pred, y)
        return loss, pred
