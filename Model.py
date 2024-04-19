import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 0)
                    nn.init.zeros_(m.bias)
                    nn.init.zeros_(m.weight)


            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
                nn.init.constant_(m.running_var, 0)
                nn.init.zeros_(m.running_var)
                nn.init.zeros_(m.running_mean)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
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

    def loss(self, x, y):
        return self.loss(x, y)

    def accuracy(self, x, y):
        pred = self.predict(x)
        return self.accuracy(pred, y)

    def evaluate(self, x, y):
        pred = self.predict(x)
        return self.accuracy(pred, y)

    def train_step(self, x, y):
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